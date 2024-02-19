import copy
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data

country_mapping = pd.read_csv("../data_files/ogbn_mag/papers.csv")['final']  # get mapping index \to country
labels_stat = pd.read_csv("../data_files/ogbn_mag/label_stat.csv")["label"].tolist()
label_mapping = {j: i for i, j in enumerate(labels_stat)}


def rewrite_tensor(tensor, mappings):
    tensor = copy.deepcopy(tensor)
    rewrite_function = lambda x: mappings[x]
    rewritten_tensor = tensor.apply_(rewrite_function)
    return rewritten_tensor


class Gen_Separate_Mag_Graph:
    """
    generate separate graph from obgn-mag given its country code and label-merging manner.
    """

    def __init__(self, Dataset, country_code, num_labels, dir_="../data_files/ogbn_mag/"):
        """

        :param Dataset: Ogbn-mag dataset
        :param num_labels: Select Top-K-occurrence labels, and merge all other labels into one.
        """
        self.data = Dataset.data
        self.nodes_emb_all = self.data["x_dict"]["paper"]
        self.edges_idx_all = self.data.edge_index_dict[('paper', 'cites', 'paper')]
        self.label_all = self.data["y_dict"]["paper"].squeeze()
        self.country = country_code
        self.num_labels = num_labels
        self.in_nodes = None  # nodes in the separate graph
        self.num_nodes = 0
        self.nodes_emb = None
        self.edges_idx = None
        self.labels = None
        self.download_path = dir_ + f"{country_code}_labels_{num_labels}.pt"
        # node embedding and edge index after re-indexing

    def parse_edges(self):
        """
        we choose an edge if its two connected nodes both belongs to the country.
        :return: parsed edge index (through re-indexing)
        """
        citing, cited = self.edges_idx_all
        citing, cited = citing.tolist(), cited.tolist()
        citing_within_spr, cited_within_spr = [], []
        for i, j in zip(citing, cited):
            if country_mapping[i] == self.country and country_mapping[j] == self.country:
                citing_within_spr.append(i)
                cited_within_spr.append(j)
        citing_within_spr, cited_within_spr = torch.tensor(citing_within_spr), torch.tensor(cited_within_spr)
        init_edges = torch.stack((citing_within_spr, cited_within_spr), dim=0)

        self.in_nodes = torch.unique(init_edges)  # Nodes included in this graph
        self.num_nodes = self.in_nodes.shape[0]
        self.nodes_emb = self.nodes_emb_all[self.in_nodes]
        re_index = {j: i for i, j in enumerate(self.in_nodes.tolist())}
        self.edges_idx = rewrite_tensor(init_edges, re_index)

    def merge_labels(self):
        label_mapping_ = {k: min(v, self.num_labels - 1) for k, v in label_mapping.items()}
        in_labels = self.label_all[self.in_nodes]
        self.labels = rewrite_tensor(in_labels, label_mapping_)

    def generate(self):
        self.parse_edges()
        self.merge_labels()
        Graph = Data(x=self.nodes_emb, edge_index=self.edges_idx, y=self.labels)
        torch.save(Graph, self.download_path)


if __name__ == '__main__':
    dataset = PygNodePropPredDataset(name="ogbn-mag")
    gen_dataset = Gen_Separate_Mag_Graph(Dataset=dataset, country_code="DE", num_labels=20)
    gen_dataset.generate()
