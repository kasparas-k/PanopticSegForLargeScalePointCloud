import numpy as np
import torch
import torch
from sklearn.cluster import MeanShift
from .mean_shift_euc_gpu import MeanShiftEuc
import time
import multiprocessing
from multiprocessing import Process
from functools import partial
def meanshift_cluster(prediction, bandwidth):
    bandwidth = bandwidth #0.6
    #ms = MeanShift(bandwidth=bandwidth,bin_seeding=True, n_jobs=-1)
    ms = MeanShiftEuc(bandwidth=bandwidth) #, n_jobs=-1)
    #print ('Mean shift clustering, might take some time ...')
    ms.fit(prediction)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_ 	
    #num_clusters = cluster_centers.shape[0]
        
    return torch.from_numpy(labels)
        
def cluster_loop(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, low, high, loop_num):
    
    #t = time.time()
    all_clusters = []
    cluster_type = []
    final_result = []
    local_logits = []
    cluster_type_loop = []
    
    embed_logits_logits_u = embed_logits_logits_u.detach()
    unique_in_batch = unique_in_batch.detach()
    label_batch = label_batch.detach()
    local_ind = local_ind.detach()
    pick_num  = 5 #np.random.randint(low=low,high=high+1,size=loop_num)
    for loop_i in range(loop_num):
        #feature_choose = np.random.choice(embed_logits_logits_u.shape[-1], pick_num[loop_i], replace=False)
        #feature_choose = torch.multinomial(torch.ones(embed_logits_logits_u.shape[-1]), pick_num[loop_i], replacement=False, out=None)
        feature_choose = torch.multinomial(torch.ones(embed_logits_logits_u.shape[-1]), pick_num, replacement=False, out=None)
        #print('type %d feature choose:' % (loop_i))
        #print(feature_choose)
        embed_logits_logits_typei = embed_logits_logits_u[:,feature_choose]
        for s in unique_in_batch:
            batch_mask = label_batch == s
            if torch.sum(batch_mask)>5:
                sampleInBatch_local_ind = local_ind[batch_mask]
                local_logits.append(sampleInBatch_local_ind)
                sample_embed_logits = embed_logits_logits_typei[batch_mask]
                #sample_embed_logits = torch.nn.functional.normalize(sample_embed_logits, dim=0)
                all_clusters.append(sample_embed_logits.detach())
                cluster_type_loop.append(loop_i)
    
    results = []
    for i, cl in enumerate(all_clusters):
        start = time.process_time()
        results.append(meanshift_cluster(cl))
    for i in range(len(results)):
        pre_ins_labels_embed = results[i]
        sampleInBatch_local_ind = local_logits[i]
        loop_i_ = cluster_type_loop[i]
        unique_preInslabels = torch.unique(pre_ins_labels_embed)
        for l in unique_preInslabels:
            if l == -1:
                continue
            label_mask_l = pre_ins_labels_embed == l
            final_result.append(sampleInBatch_local_ind[label_mask_l])
            cluster_type.append(loop_i_)
    
    #print("total time",time.time()-t)
    return final_result, cluster_type

def cluster_single(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, type, bandwidth):
    #t = time.time()
    all_clusters = []
    cluster_type = []
    final_result = []
    local_logits = []
    
    embed_logits_logits_u = embed_logits_logits_u.cpu().detach()
    unique_in_batch = unique_in_batch.cpu().detach()
    label_batch = label_batch.cpu().detach()
    local_ind = local_ind.cpu().detach()

    for s in unique_in_batch:
        batch_mask = label_batch == s
        if torch.sum(batch_mask)>3:
            sampleInBatch_local_ind = local_ind[batch_mask]
            local_logits.append(sampleInBatch_local_ind)
            sample_embed_logits = embed_logits_logits_u[batch_mask]
            #meanshift
            #sample_embed_logits = torch.nn.functional.normalize(sample_embed_logits, dim=0)
            all_clusters.append(sample_embed_logits.cpu().detach().numpy())
            #normalize(sample_embed_logits, axis=0)
    
    partial_meanshift_cluster = partial(meanshift_cluster, bandwidth=bandwidth)
    results = []
    for i, cl in enumerate(all_clusters):
        start = time.process_time()
        results.append(partial_meanshift_cluster(cl))
    for i in range(len(results)):
        pre_ins_labels_embed = results[i]
        sampleInBatch_local_ind = local_logits[i]
        unique_preInslabels = torch.unique(pre_ins_labels_embed)
        for l in unique_preInslabels:
            if l == -1:
                continue
            label_mask_l = pre_ins_labels_embed == l
            final_result.append(sampleInBatch_local_ind[label_mask_l])
            cluster_type.append(type)        
        
        #pre_ins_labels_embed = hdbscan_cluster(sample_embed_logits)
        #unique_preInslabels = torch.unique(pre_ins_labels_embed)
        #for l in unique_preInslabels:
        #    if l == -1:
        #        continue
        #    label_mask_l = pre_ins_labels_embed == l
        #    all_clusters.append(sampleInBatch_local_ind[label_mask_l])
        #    cluster_type.append(type)
                
    #print("total time",time.time()-t)
    return final_result, cluster_type

if __name__ == "__main__":
    embed_logits_logits_u = torch.load("embed_logits_logits_u.pt").cpu()
    label_batch = torch.load("label_batch.pt").cpu()
    local_ind = torch.load("local_ind.pt").cpu()
    unique_in_batch = torch.load("unique_in_batch.pt").cpu()
    print(np.shape(embed_logits_logits_u))
    print(np.shape(label_batch))
    print(np.shape(local_ind))
    print(np.shape(unique_in_batch))
    t = time.time()
    for i in range(4):
        cluster_loop(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, low=3, high=5, loop_num=10)
    print("total time",time.time()-t)
