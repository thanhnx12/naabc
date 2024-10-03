import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from config import Config
import torch.nn.functional as F


from sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment
from encoder_mixup import EncodingModel
import pickle as pkl
from add_loss import InClusterLoss, ClusterLoss, MultipleNegativesRankingLoss, TripletLoss


class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.relation_embeddings = pkl.load(open(config.relation_embedding, 'rb')) # {rel: np array embedding}
        
    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist
    
    def _edist_cosine(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        # x2 = torch.tensor(x2)

        x1 = F.normalize(x1, p=2, dim=1)  # Normalize along the last dimension (H)
        x2 = F.normalize(x2, p=2, dim=1)
        # Compute cosine similarity
        dist = torch.matmul(x1, x2.T)  # (B, H) x (H, N) --> (B, N)
        
        # Convert cosine similarity to cosine distance
        dist = 1 - dist  # Since cosine similarity ranges from -1 to 1

        return dist
    


    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        # proto = memory mean
        # rel_proto = mem_feas.mean(0)
        # proto = all mean
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas
        # return mem_set, features, rel_proto
        
    # def get_augment_data_label(self, instance, labels):
    #     max_len = self.config.max_length
    #     batch_size, dim = instance['ids'].shape

    #     augmented_ids = []
    #     augmented_masks = []
    #     augmented_labels = []
    #     label_first = []
    #     label_second = []
        
    #     random_list_j = []

    #     if batch_size < 4:
    #         # bs 4 always has positive pair 
    #         random_list_i = random.sample(range(0, batch_size), min(4, batch_size))
    #         random_list_j = random.sample(range(0, batch_size), min(4, batch_size))
    #     else:
    #         # Create a dictionary of positive pairs: key is the index, value is a list of indices with the same label
    #         positive_dict = {}
    #         for idx, label in enumerate(labels):
    #             label = label.item()
    #             if label not in positive_dict:
    #                 positive_dict[label] = []
    #             positive_dict[label].append(idx)

    #         # Convert the positive dict into a form where key is the index and value is a list of positive pairs
    #         positive_pairs = {}
    #         for indices in positive_dict.values():
    #             for i in indices:
    #                 positive_pairs[i] = [j for j in indices if j != i]

    #         # Randomly select 4 indices for i
    #         random.seed(42)
    #         random_list_i = random.sample(range(0, batch_size), min(4, batch_size))

    #         # Ensure at least one positive pair in random_list_j
    #         random_list_j = []

    #         for i in random_list_i:
    #             if i in positive_pairs and positive_pairs[i]:
    #                 # Find a positive j that is different from i
    #                 j = random.choice(positive_pairs[i])
    #             else:
    #                 # If no positive available, set j to i
    #                 j = i
    #             random_list_j.append(j)

    #     # if batch_size < 4:
    #     #     # bs 4 always has positive pair 
    #     #     random_list_i = random.sample(range(0, batch_size), min(4, batch_size))
    #     #     random_list_j = random.sample(range(0, batch_size), min(4, batch_size))
    #     # else:
    #     #     random.seed(42)
    #     #     random_list_i = random.sample(range(0, batch_size), 4)

    #     #     for i in random_list_i:
    #     #         if len(random_list_j) != 0:
    #     #             break
    #     #         for j in range(batch_size):
    #     #             if i!=j and labels[i]==labels[j]:
    #     #                 random_list_j.append(j)

    #     #     if len(random_list_j) == 0:
    #     #         random_list_j.append(random_list_i[0])
                        
    #     #     remaining_elements = list(set(range(0, batch_size)) - set(random_list_i))
            
    #     #     if len(remaining_elements) >= 3:
    #     #         random_list_j.extend(random.sample(remaining_elements, 3))
    #     #     else:
    #     #         random_list_j.extend(random.sample(range(0, batch_size), 3))

       
   
    #     for i in random_list_i:
    #         for j in random_list_j:
    #             # Filter 'ids' using the corresponding 'mask' to remove zero padding
    #             ids1 = instance['ids'][i][instance['mask'][i] != 0]  # Remove padding from the first sequence
    #             ids2 = instance['ids'][j][instance['mask'][j] != 0]  # Remove padding from the second sequence

    #             # Concatenate the filtered sequences
    #             combined_ids = torch.cat((ids1, ids2)).to(config.device)

    #             # Truncate the concatenated sequence if it exceeds max_len - 1 and add [102] at the end
    #             if len(combined_ids) > max_len - 1:
    #                 combined_ids = combined_ids[:max_len - 1]
    #                 combined_ids = torch.cat((combined_ids, torch.tensor([102], dtype=combined_ids.dtype).to(config.device))).to(config.device)

    #                 # Calculate the mask: 1 for valid positions, 0 for padding
    #             combined_mask = torch.ones_like(combined_ids, dtype=torch.float).to(config.device)

    #             # Pad with zeros if the sequence is shorter than max_len
    #             if len(combined_ids) < max_len:
    #                 padding_length = max_len - len(combined_ids)
    #                 padding = torch.zeros(padding_length, dtype=combined_ids.dtype).to(config.device)
    #                 combined_ids = torch.cat((combined_ids, padding)).to(config.device)

    #                 # Update the mask with zeros for padded positions
    #                 combined_mask = torch.cat((combined_mask, torch.zeros(padding_length, dtype=torch.float).to(config.device)))

    #             augmented_ids.append(combined_ids)
    #             augmented_masks.append(combined_mask)

    #             # Construct the label pairs
    #             new_label = torch.tensor([labels[i], labels[j]])
    #             augmented_labels.append(new_label)
    #             label_first.append(labels[i])
    #             label_second.append(labels[j])

    #     # Convert the lists into tensors
    #     augmented_data = {
    #         'ids': torch.stack(augmented_ids),
    #         'mask': torch.stack(augmented_masks)
    #     }
    #     augmented_labels = torch.stack(augmented_labels)
    #     label_first = torch.tensor(label_first)
    #     label_second = torch.tensor(label_second)

    #     return augmented_data, augmented_labels, label_first, label_second


    def get_augment_data_label(self, instance, labels):
        max_len = self.config.max_length
        batch_size, dim = instance['ids'].shape

        augmented_ids = []
        augmented_masks = []
        augmented_labels = []
        label_first = []
        label_second = []

        if batch_size < 4:
            random_list_i = random.sample(range(0, batch_size), min(4, batch_size))
            random_list_j = random.sample(range(0, batch_size), min(4, batch_size))
        else:
            random.seed(42)
            random_list_i = random.sample(range(0, batch_size), 4)
            remaining_elements = list(set(range(0, batch_size)) - set(random_list_i))
            
            if len(remaining_elements) >= 4:
                random_list_j = random.sample(remaining_elements, 4)
            else:
                random_list_j = random.sample(range(0, batch_size), 4)
    
        for i in random_list_i:
            for j in random_list_j:
                # Filter 'ids' using the corresponding 'mask' to remove zero padding
                ids1 = instance['ids'][i][instance['mask'][i] != 0]  # Remove padding from the first sequence
                ids2 = instance['ids'][j][instance['mask'][j] != 0]  # Remove padding from the second sequence

                # Concatenate the filtered sequences
                combined_ids = torch.cat((ids1, ids2)).to(config.device)

                # Truncate the concatenated sequence if it exceeds max_len - 1 and add [102] at the end
                if len(combined_ids) > max_len - 1:
                    combined_ids = combined_ids[:max_len - 1]
                    combined_ids = torch.cat((combined_ids, torch.tensor([102], dtype=combined_ids.dtype).to(config.device))).to(config.device)

                    # Calculate the mask: 1 for valid positions, 0 for padding
                combined_mask = torch.ones_like(combined_ids, dtype=torch.float).to(config.device)

                # Pad with zeros if the sequence is shorter than max_len
                if len(combined_ids) < max_len:
                    padding_length = max_len - len(combined_ids)
                    padding = torch.zeros(padding_length, dtype=combined_ids.dtype).to(config.device)
                    combined_ids = torch.cat((combined_ids, padding)).to(config.device)

                    # Update the mask with zeros for padded positions
                    combined_mask = torch.cat((combined_mask, torch.zeros(padding_length, dtype=torch.float).to(config.device)))

                augmented_ids.append(combined_ids)
                augmented_masks.append(combined_mask)

                # Construct the label pairs
                new_label = torch.tensor([labels[i], labels[j]])
                augmented_labels.append(new_label)
                label_first.append(labels[i])
                label_second.append(labels[j])

        # Convert the lists into tensors
        augmented_data = {
            'ids': torch.stack(augmented_ids),
            'mask': torch.stack(augmented_masks)
        }
        augmented_labels = torch.stack(augmented_labels)
        label_first = torch.tensor(label_first)
        label_second = torch.tensor(label_second)

        return augmented_data, augmented_labels, label_first, label_second
    def train_model(self, encoder, training_data, is_memory=False, relationid2cluster=None , seen_relation_embeddings = None, relationid2cluster_centroids = None ):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        optimizer = optim.Adam(params=encoder.parameters(), lr=self.config.lr)
        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch
        
        in_cluster_loss = InClusterLoss()
        cluster_loss = ClusterLoss()    
        triplet = TripletLoss()
        
        seen_relation_embeddings = torch.from_numpy(seen_relation_embeddings).to(self.config.device)

        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)
                hidden = encoder(instance)
                
                # loss1 = in_cluster_loss(hidden, labels, relationid2cluster)

                # relationid2cluster_centroids IS A DICT KEY IS relation , value is embedding of centroids 

                # self.relation_embeddings is a dict key is name_relationn ,value is embedding ex :'main subject': array([ 0.01104855,  0.00419397,  0.03440297, ..., -0.00340303,
                
                # centroids_current chính là cái embedding des_cription của label đó

                # 

                centroids_current = []
                cluster_centroids = []
                for label in labels:
                    # centroids_current.append(torch.from_numpy(self.relation_embeddings[self.id2rel[label.item()]][:768]))
                    centroids_current.append(torch.from_numpy(self.relation_embeddings[self.id2rel[label.item()]]))

                    cluster_centroids.append(torch.from_numpy(relationid2cluster_centroids[label.item()]))
                centroids_current = torch.stack(centroids_current, dim = 0).to(self.config.device)
                cluster_centroids = torch.stack(cluster_centroids, dim = 0).to(self.config.device)
    
                
                centroids_current = centroids_current.squeeze(1)
              

                nearest_cluster_centroids = []
                for hid in hidden:
                    # Compute cosine similarities between the hidden vector and all seen relation embeddings
                    # cos_similarities = torch.nn.functional.cosine_similarity(hid.unsqueeze(0), seen_relation_embeddings, dim=1)4
                    cos_similarities = torch.cdist(hid.unsqueeze(0), seen_relation_embeddings, p = 2)[0]
                    # Get the indices of the top 2 maximum cosine similarities
                    top2_similarities, top2_indices = torch.topk(cos_similarities, k=2, dim=0)
                    # print(torch.argmax(cos_similarities))
                    # print(top2_indices)
                    # Gather only the top 2 centroids
                    top2_centroids = seen_relation_embeddings[top2_indices[1].item()]
                    # print(top2_indices)
                    # print(top2_indices[1].item())
                    
                    # Append the top 2 centroids to the list
                    nearest_cluster_centroids.append(top2_centroids)

                # Initialize loss accumulators
                    # total_loss2 = 0.0
                    # total_loss3 = 0.0

                    # # Iterate over the batch
                    # for i in range(hidden.shape[0]):
                    #     # Compute losses for the i-th item in the batch
                    #     loss2 = triplet(hidden[i], centroids_current[i], cluster_centroids[i])
                    #     loss3 = triplet(hidden[i], centroids_current[i], nearest_cluster_centroids[i])
                        
                    #     # Accumulate the losses
                    #     total_loss2 += loss2
                    #     total_loss3 += loss3

                    # # Compute the mean losses
                    # mean_loss2 = total_loss2 / hidden.shape[0]
                    # mean_loss3 = total_loss3 / hidden.shape[0]

                    # # Optionally, you can combine them if needed
                    # # For example, if you want to take a weighted average or just sum them up
                    # combined_loss = mean_loss2 + mean_loss3



                nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim=0)

                # loss2 = triplet(hidden, centroids_current,  cluster_centroids)

                # loss3 = triplet(hidden, centroids_current, nearest_cluster_centroids)
                loss2 = self.moment.contrastive_loss_des(centroids_current, hidden, labels)
                # loss3 = self.moment.contrastive_loss_des(hidden, cluster_centroids, labels)
              

                # loss2 = cluster_loss(hidden, labels, relationid2cluster)
                

                # seen_relation_embeddings = np.stack(seen_relation_embeddings, axis=0) # (N, H)




                # augmented_instance, augmented_labels, label_first, label_second = self.get_augment_data_label(instance, labels)
         

                # mask_hidden_1, mask_hidden_2 = encoder(augmented_instance, is_augment = True)

                # n = len(label_first)
                # m = len(label_second)
                # new_matrix_labels = np.zeros((n, m), dtype=float)

                # # Fill the matrix according to the label comparison
                # for i1 in range(n):
                #     for j in range(m):
                #         if label_first[i1] == label_second[j]:
                #             new_matrix_labels[i1][j] = 1.0

                # new_matrix_labels_tensor = torch.tensor(new_matrix_labels).to(config.device)

                # loss1 = self.moment.contrastive_loss(hidden, labels, is_memory)

                
                # loss_retrieval = MultipleNegativesRankingLoss()
                # loss4 = loss_retrieval(mask_hidden_1, mask_hidden_2, new_matrix_labels_tensor)
                # loss4 = self.moment.contrastive_loss_des(mask_hidden_1, mask_hidden_2, label_first, labels2 = label_second, is_augment = True)

                # loss4 = self.moment.contrastive_loss_des(centroids_current, hidden, labels)

                # loss1 = self.moment.contrastive_loss(hidden, labels, is_memory)
                loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, centroids_current)

                # loss1 = self.moment.contrastive_loss(hidden, labels, is_memory)
                # loss4 = triplet(hidden, centroids_current,  cluster_centroids)

                loss = loss1 + loss2 
                print("Losses: ", loss1.item(), loss2.item())
                

                # loss = loss + loss2 
                # print("Losses: ", loss.item(), loss2.item())


                            
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # update moment
                # if is_memory:
                #     self.moment.update(ind, hidden.detach().cpu().data, is_memory=True)
                #     # self.moment.update_allmem(encoder)
                # else:
                #     self.moment.update(ind, hidden.detach().cpu().data, is_memory=False)

                if is_memory:
                    self.moment.update_des(ind, hidden.detach().cpu().data, centroids_current.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update_des(ind, hidden.detach().cpu().data, centroids_current.detach().cpu().data, is_memory=False)
                
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush() 
        print('')             

    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data, seen_embedding = None):
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data # place in cpu to eval

            logits = -self._edist_cosine(fea, seen_proto) # (B, N) ;N is the number of seen relations
            # logits = -self._edist_cosine(fea, seen_embedding) # (B, N) ;N is the number of seen relations


            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size
            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   '\
                .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.flush()        
        print('')
        return corrects / total

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset

    def get_n_clusters_elbow(self, embeddings):
        distortions = []
        K = range(1, embeddings.shape[0] // 2)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(embeddings)
            distortions.append(kmeanModel.inertia_)
        return np.argmin(distortions) + 1 #
    def get_cluster(self, embeddings, n_clusters):
        # Calculate the pairwise cosine distances between the embeddings
        distance_matrix = cosine_distances(embeddings)

        # Perform hierarchical clustering
        # You can adjust the number of clusters by changing the `n_clusters` parameter
        
        # old 
        # clustering_model = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
        # clusters = clustering_model.fit_predict(distance_matrix)

        clustering_model = AgglomerativeClustering(n_clusters=5, metric = 'euclidean', linkage='average')
        clusters = clustering_model.fit_predict(embeddings)

        return clusters
    
    def get_cluster_and_centroids(self, embeddings, n_clusters):
        # Calculate the pairwise cosine distances between the embeddings
        distance_matrix = cosine_distances(embeddings)

        # Perform hierarchical clustering with cosine distance
        clustering_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='average')
        # clustering_model = AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average", distance_threshold=0.5)

        clusters = clustering_model.fit_predict(embeddings)

        # Initialize a list to hold the centroids for each cluster
        centroids = []

        # Calculate the centroid (mean) of each cluster
        for cluster_id in range(n_clusters):
            # Get the embeddings belonging to the current cluster
            cluster_embeddings = embeddings[clusters == cluster_id]
            
            # Calculate the centroid by taking the mean of the embeddings in the cluster
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids.append(centroid)

        # Return both the clusters and their centroids
        return clusters, np.array(centroids)

    def train(self):
        # sampler 
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = EncodingModel(self.config)

        # step is continual task number
        cur_acc, total_acc = [], []
        cur_acc_num, total_acc_num = [], []
        memory_samples = {}
        data_generation = []
        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations) in enumerate(sampler):
        
            # Initialization
            self.moment = Moment(self.config)
                
            # Clustering seen relations
            seen_relation_embeddings = []
            self.seen_relation_embeddings_infer = {}
            for rel in seen_relations:
                seen_relation_embeddings.append(self.relation_embeddings[rel])
                self.seen_relation_embeddings_infer[rel] = self.relation_embeddings[rel]
                

            seen_relation_embeddings = np.stack(seen_relation_embeddings, axis=0) # (N, H)
            seen_relation_embeddings = seen_relation_embeddings.squeeze(axis=1)
            # print(len(seen_relation_embeddings))
            # print(len(seen_relation_embeddings[0]))
            # print(len(seen_relation_embeddings[0][0]))
            # seen_relation_embeddings = seen_relation_embeddings[:, :768]

            num_clusters = self.get_n_clusters_elbow(seen_relation_embeddings)
            print("Clustering into ", num_clusters, " clusters")
            clusters, clusters_centroids = self.get_cluster_and_centroids(seen_relation_embeddings, num_clusters) # array (N)
            print("Clusters: ", clusters)
            relationid2cluster = {}
            relationid2cluster_centroids = {}
            for index, rel in enumerate(seen_relations):
                relationid2cluster[self.rel2id[rel]] = clusters[index]
                relationid2cluster_centroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]

                
                

            # Train current task
            training_data_initialize = []
            for rel in current_relations:
                training_data_initialize += training_data[rel]
            self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
            # if step > 0:
            self.train_model(encoder, training_data_initialize, relationid2cluster=relationid2cluster, seen_relation_embeddings = seen_relation_embeddings, relationid2cluster_centroids = relationid2cluster_centroids )

            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])

            # Data gen
            # if self.config.gen == 1:
            #     gen_text = []
            #     for rel in current_relations:
            #         for sample in memory_samples[rel]:
            #             sample_text = self._get_sample_text(self.config.training_data, sample['index'])
            #             gen_samples = gen_data(self.r2desc, self.rel2id, sample_text, self.config.num_gen, self.config.gpt_temp, self.config.key)
            #             gen_text += gen_samples
            #     for sample in gen_text:
            #         data_generation.append(sampler.tokenize(sample))
                    
            # Train memory
            if step > 0:
                memory_data_initialize = []
                for rel in seen_relations:
                    memory_data_initialize += memory_samples[rel]
                memory_data_initialize += data_generation
                self.moment.init_moment(encoder, memory_data_initialize, is_memory=True) 
                # if step > 0:
                self.train_model(encoder, memory_data_initialize, is_memory=True, relationid2cluster=relationid2cluster, seen_relation_embeddings = seen_relation_embeddings, relationid2cluster_centroids = relationid2cluster_centroids)

            # Update proto
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            # Eval current task and history task
            test_data_initialize_cur, test_data_initialize_seen = [], []
            for rel in current_relations:
                test_data_initialize_cur += test_data[rel]
            for rel in seen_relations:
                test_data_initialize_seen += historic_test_data[rel]
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])
            ac1 = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_cur, seen_embedding = seen_relation_embeddings)
            ac2 = self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data_initialize_seen, seen_embedding = seen_relation_embeddings)
            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            cur_acc.append('{:.4f}'.format(ac1))
            total_acc.append('{:.4f}'.format(ac2))
            print('cur_acc: ', cur_acc)
            print('his_acc: ', total_acc)

        torch.cuda.empty_cache()
        return total_acc_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen

    # config 
    print('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print('#############params############')

    if config.task_name == 'FewRel':
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        config.relation_embedding = './data/CFRLFewRel/fewrel_embeddings.pkl'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.relation_embedding = './data/CFRLTacred/relation_embeddings_1.pkl'
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'        

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    acc_list = []
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        print('seed: ', config.seed)
        manager = Manager(config)
        acc = manager.train()
        acc_list.append(acc)
        torch.cuda.empty_cache()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    print('----------END')
    print('his_acc mean: ', np.around(ave, 4))



            
        
            
            


