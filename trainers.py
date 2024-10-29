from tenpcs.models.functional import integrate
from tenpcs.layers.sum_product import CollapsedCPLayer

import numpy as np
import functools
import torch
import time
import os

print = functools.partial(print, flush=True)

from utils import eval_ll, ll2bpd


def log_string(
    pc,
    train_step: int,
    dataset_str: str,
    curr_train_ll: float,
    best_valid_ll: float,
    best_test_ll: float,
    train_cycle_time: float,
    avg_batch_time: float,
    lr: float,
    device,
):
    num_features = pc.num_vars * pc.input_layer.num_channels
    curr_train_bpd = ll2bpd(curr_train_ll, num_features)
    best_valid_bpd = ll2bpd(best_valid_ll, num_features)
    best_test_bpd = ll2bpd(best_test_ll, num_features)
    print(f"step: {train_step}, {dataset_str}, lr: {lr:.5f}, train cycle completed in {train_cycle_time:.2f}s \n"
          f"curr-train LL: {curr_train_ll:.2f}, best-valid LL: {best_valid_ll:.2f}, best-test LL: {best_test_ll:.2f}\n"
          f"curr-train bpd: {curr_train_bpd:.10f}, best-valid bpd: {best_valid_bpd:.10f}, best-test bpd: {best_test_bpd:.10f}\n"
          f"bt: {int(avg_batch_time)}ms, GPU: {torch.cuda.max_memory_allocated(device=device) / 1024 ** 3:.2f} GiB\n")


def training_pic(
    pic,
    qpc,
    z_quad,
    w_quad,
    optimizer,
    scheduler,
    loss_reduction: str,
    max_train_steps: int,
    patience: int,
    min_delta: float,
    train_loader,
    valid_loader,
    test_loader,
    valid_freq: int,
    writer,
    model_dir: str
):
    assert loss_reduction in ['mean', 'sum']

    device = z_quad.device
    qpc.scope_layer.to(device)
    for i, layer in enumerate(qpc.inner_layers):
        if layer._get_name() == 'SumLayer' or qpc.bookkeeping[i][0]:
            layer.to(device)

    dataset_str = model_dir.split(os.sep)[-3]
    train_step = 1
    best_valid_ll = - np.infty
    best_test_ll = - np.infty
    patience_counter = patience
    batch_time_log = []
    tik_train = tik_train_cycle = time.time()

    print(f'Training for at most {max_train_steps} steps..\n')
    while train_step < max_train_steps + 1:

        for batch in train_loader:
            batch = batch.to(device)
            tik_batch = time.time()
            train_step += 1
            pic.parameterize_qpc(qpc=qpc, z_quad=z_quad, w_quad=w_quad)
            ll = qpc(batch).mean() if loss_reduction == 'mean' else qpc(batch).sum()
            if np.isnan(ll.item()):
                print(f"-> damn, NaN! lr: {optimizer.param_groups[0]['lr']:.5f}")
                train_step = np.inf
                break
            optimizer.zero_grad()
            (-ll).backward()
            optimizer.step()
            scheduler.step()
            batch_time_log.append((time.time() - tik_batch) * 1000)

            if train_step % valid_freq == 0:
                with torch.no_grad():
                    pic.parameterize_qpc(qpc=qpc, z_quad=z_quad, w_quad=w_quad)
                train_ll = 0  # eval_ll(qpc, train_loader)  # todo
                # writer.add_scalar("train_ll", train_ll, train_step)
                valid_ll = eval_ll(qpc, valid_loader)
                writer.add_scalar("valid_ll", valid_ll, train_step)
                if (valid_ll - min_delta) <= best_valid_ll:
                    patience_counter -= 1
                    if patience_counter == 0:
                        print("-> validation LL is not improving, early stopping")
                        train_step = np.inf
                        break
                else:
                    print("-> Saved model")
                    torch.save(pic, model_dir)
                    best_test_ll = eval_ll(qpc, test_loader)
                    writer.add_scalar("best_test_ll", best_test_ll, train_step)
                    best_valid_ll = valid_ll
                    patience_counter = patience
                log_string(
                    pc=qpc, train_step=train_step, dataset_str=dataset_str,
                    curr_train_ll=train_ll, best_valid_ll=best_valid_ll, best_test_ll=best_test_ll,
                    train_cycle_time=time.time() - tik_train_cycle, avg_batch_time=np.mean(batch_time_log),
                    lr=optimizer.param_groups[0]['lr'], device=device)
                tik_train_cycle = time.time()
                batch_time_log.clear()
                writer.flush()
            if train_step > max_train_steps + 1:
                break

    train_time = time.time() - tik_train
    print(f'Overall training time: {train_time:.2f}s')


def training_pc(
    pc,
    optimizer,
    scheduler,
    loss_reduction: str,
    max_train_steps: int,
    patience: int,
    min_delta: float,
    train_loader,
    valid_loader,
    test_loader,
    valid_freq: int,
    writer,
    model_dir: str
):
    assert loss_reduction in ['mean', 'sum']
    dataset_str = model_dir.split(os.sep)[-3]
    train_step = 1
    best_valid_ll = - np.infty
    best_test_ll = - np.infty
    patience_counter = patience
    batch_time_log = []
    tik_train = tik_train_cycle = time.time()

    device = pc.input_layer.params.param.device
    pc_pf = integrate(pc)
    sqrt_eps = np.sqrt(torch.finfo(torch.get_default_dtype()).tiny)

    print(f'Training for at most {max_train_steps} steps..\n')
    while train_step < max_train_steps + 1:

        for batch in train_loader:
            batch = batch.to(device)
            tik_batch = time.time()
            train_step += 1
            ll = (pc(batch) - pc_pf(None)).mean() if loss_reduction == 'mean' else (pc(batch) - pc_pf(None)).sum()
            if np.isnan(ll.item()):
                print(f"-> damn, NaN! lr: {optimizer.param_groups[0]['lr']:.5f}")
                train_step = np.inf
                break
            optimizer.zero_grad()
            (-ll).backward()
            optimizer.step()
            scheduler.step()
            batch_time_log.append((time.time() - tik_batch) * 1000)

            for layer in pc.inner_layers:
                if isinstance(layer, CollapsedCPLayer):
                    layer.params_in().data.clamp_(min=sqrt_eps)
                else:  # SumLayer, TuckerLayer
                    layer.params().data.clamp_(min=sqrt_eps)

            if train_step % valid_freq == 0:
                train_ll = 0  # eval_ll(pc, train_loader)  # todo
                # writer.add_scalar("train_ll", train_ll, train_step)
                valid_ll = eval_ll(pc, valid_loader)
                writer.add_scalar("valid_ll", valid_ll, train_step)
                if (valid_ll - min_delta) <= best_valid_ll:
                    patience_counter -= 1
                    if patience_counter == 0:
                        print("-> validation LL is not improving, early stopping")
                        train_step = np.inf
                        break
                else:
                    print("-> Saved model")
                    torch.save(pc, model_dir)
                    best_test_ll = eval_ll(pc, test_loader)
                    writer.add_scalar("best_test_ll", best_test_ll, train_step)
                    best_valid_ll = valid_ll
                    patience_counter = patience
                log_string(
                    pc=pc, train_step=train_step, dataset_str=dataset_str,
                    curr_train_ll=train_ll, best_valid_ll=best_valid_ll, best_test_ll=best_test_ll,
                    train_cycle_time=time.time() - tik_train_cycle, avg_batch_time=np.mean(batch_time_log),
                    lr=optimizer.param_groups[0]['lr'], device=device)
                tik_train_cycle = time.time()
                batch_time_log.clear()
                writer.flush()
            if train_step > max_train_steps + 1:
                break

    train_time = time.time() - tik_train
    print(f'Overall training time: {train_time:.2f}s')


def test_pc(
    pc,
    train_loader,
    valid_loader,
    test_loader,
):
    results = {
        'train_ll': eval_ll(pc, train_loader),
        'valid_ll': eval_ll(pc, valid_loader),
        'test_ll': eval_ll(pc, test_loader)}
    num_features = pc.num_vars * pc.input_layer.num_channels
    results['train_bpd'] = ll2bpd(results['train_ll'], num_features)
    results['valid_bpd'] = ll2bpd(results['valid_ll'], num_features)
    results['test_bpd'] = ll2bpd(results['test_ll'], num_features)
    print(f"train ({results['train_ll']:.10f}, {results['train_bpd']:.10f})")
    print(f"valid ({results['valid_ll']:.10f}, {results['valid_bpd']:.10f})")
    print(f"test ({results['test_ll']:.10f}, {results['test_bpd']:.10f})")
    return results
