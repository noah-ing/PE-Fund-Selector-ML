"""
Graph Neural Network for company relationship mapping
Uses GraphSAGE to capture hidden supply chain and partnership signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.utils import negative_sampling
import networkx as nx
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphData:
    """Container for graph data"""
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: Optional[torch.Tensor] = None
    node_labels: Optional[torch.Tensor] = None
    company_names: Optional[List[str]] = None
    
    def to_pyg_data(self) -> Data:
        """Convert to PyTorch Geometric Data object"""
        data = Data(
            x=self.node_features,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.node_labels
        )
        return data

class GraphSAGE(nn.Module):
    """
    GraphSAGE model for company relationship learning
    Captures multi-hop neighborhood information
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: List[int] = [128, 64, 32],
                 output_dim: int = 16,
                 dropout: float = 0.2,
                 aggregator: str = 'mean'):
        super(GraphSAGE, self).__init__()
        
        self.dropout = dropout
        
        # Build SAGE layers
        self.sage_layers = nn.ModuleList()
        
        # First layer
        self.sage_layers.append(
            SAGEConv(input_dim, hidden_dims[0], aggr=aggregator)
        )
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.sage_layers.append(
                SAGEConv(hidden_dims[i], hidden_dims[i + 1], aggr=aggregator)
            )
        
        # Output layer
        self.sage_layers.append(
            SAGEConv(hidden_dims[-1], output_dim, aggr=aggregator)
        )
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_dims + [output_dim]
        ])
        
        # Final MLP for scoring
        self.scorer = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass through GraphSAGE
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes (for batched graphs)
        
        Returns:
            Node embeddings and graph-level scores
        """
        # Apply SAGE layers with batch norm and dropout
        for i, (sage, bn) in enumerate(zip(self.sage_layers, self.batch_norms)):
            x = sage(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            
            if i < len(self.sage_layers) - 1:  # No dropout on last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Store node embeddings
        node_embeddings = x
        
        # Graph-level pooling
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)
        
        # Generate acquisition score
        score = self.scorer(graph_embedding)
        
        return node_embeddings, score

class CompanyGraphNetwork:
    """
    Complete GNN pipeline for company relationship analysis
    Identifies hidden connections and acquisition opportunities
    """
    
    def __init__(self, 
                 feature_dim: int = 7,
                 hidden_dims: List[int] = [128, 64, 32],
                 embedding_dim: int = 16):
        
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        
        # Initialize GraphSAGE model
        self.model = GraphSAGE(
            input_dim=feature_dim,
            hidden_dims=hidden_dims,
            output_dim=embedding_dim
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
        
        logger.info(f"GNN model initialized on {self.device}")
    
    def build_company_graph(self, 
                           companies_data: List[Dict],
                           relationships: Optional[List[Tuple[str, str]]] = None) -> nx.Graph:
        """
        Build NetworkX graph from company data
        
        Args:
            companies_data: List of company data dicts with features
            relationships: Optional list of (company1, company2) tuples
        
        Returns:
            NetworkX graph with company nodes and relationships
        """
        G = nx.Graph()
        
        # Add nodes with features
        for i, company in enumerate(companies_data):
            node_features = self._extract_node_features(company)
            G.add_node(
                company['company_name'],
                features=node_features,
                sector=company.get('sector', 'Unknown'),
                ticker=company.get('ticker', ''),
                index=i
            )
        
        # Add edges (relationships)
        if relationships:
            G.add_edges_from(relationships)
        else:
            # Infer relationships from sector and metrics similarity
            self._infer_relationships(G, companies_data)
        
        logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def _extract_node_features(self, company_data: Dict) -> np.ndarray:
        """Extract feature vector from company data"""
        # These should match the features from MLPipeline.export_features()
        features = [
            company_data.get('sentiment_score', 0.5),
            company_data.get('acquisition_probability', 0.0),
            company_data.get('financial_health_score', 0.5),
            company_data.get('growth_potential', 0.5),
            company_data.get('market_volatility', 0.5),
            company_data.get('news_momentum', 0.0),
            company_data.get('zero_shot_confidence', 0.5)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _infer_relationships(self, G: nx.Graph, companies_data: List[Dict]):
        """
        Infer relationships based on sector, supply chain patterns, and similarity
        """
        nodes = list(G.nodes())
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
                # Same sector relationships
                if G.nodes[node1]['sector'] == G.nodes[node2]['sector']:
                    if np.random.random() < 0.3:  # 30% chance of connection
                        G.add_edge(node1, node2, weight=0.5, type='sector')
                
                # Feature similarity relationships
                features1 = G.nodes[node1]['features']
                features2 = G.nodes[node2]['features']
                
                similarity = self._compute_similarity(features1, features2)
                if similarity > 0.8:  # High similarity threshold
                    G.add_edge(node1, node2, weight=similarity, type='similarity')
    
    def _compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity between feature vectors"""
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def graph_to_pyg(self, G: nx.Graph) -> GraphData:
        """
        Convert NetworkX graph to PyTorch Geometric format
        
        Returns:
            GraphData object ready for GNN processing
        """
        # Extract node features
        node_features = []
        company_names = []
        
        for node in G.nodes():
            node_features.append(G.nodes[node]['features'])
            company_names.append(node)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # Normalize features
        if len(node_features) > 1:
            node_features = self.scaler.fit_transform(node_features)
        
        # Convert to tensor
        x = torch.tensor(node_features, dtype=torch.float32)
        
        # Build edge index
        edge_list = []
        edge_weights = []
        
        node_to_idx = {node: i for i, node in enumerate(G.nodes())}
        
        for edge in G.edges(data=True):
            src = node_to_idx[edge[0]]
            dst = node_to_idx[edge[1]]
            weight = edge[2].get('weight', 1.0)
            
            # Add both directions for undirected graph
            edge_list.extend([[src, dst], [dst, src]])
            edge_weights.extend([weight, weight])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        
        return GraphData(
            node_features=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            company_names=company_names
        )
    
    def train_step(self, data: GraphData, labels: torch.Tensor) -> float:
        """
        Single training step
        
        Args:
            data: GraphData object
            labels: Target labels for nodes or graph
        
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        x = data.node_features.to(self.device)
        edge_index = data.edge_index.to(self.device)
        labels = labels.to(self.device)
        
        # Forward pass
        embeddings, scores = self.model(x, edge_index)
        
        # Compute loss
        loss = self.criterion(scores.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict(self, data: GraphData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for companies
        
        Returns:
            Tuple of (node_embeddings, acquisition_scores)
        """
        self.model.eval()
        
        with torch.no_grad():
            x = data.node_features.to(self.device)
            edge_index = data.edge_index.to(self.device)
            
            embeddings, scores = self.model(x, edge_index)
            
            embeddings = embeddings.cpu().numpy()
            scores = scores.cpu().numpy()
        
        return embeddings, scores
    
    def rank_companies(self, 
                      companies_data: List[Dict],
                      relationships: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Rank companies using GNN embeddings and graph structure
        
        Args:
            companies_data: List of company data with features
            relationships: Optional explicit relationships
        
        Returns:
            DataFrame with companies ranked by GNN score
        """
        # Build graph
        G = self.build_company_graph(companies_data, relationships)
        
        # Convert to PyG format
        graph_data = self.graph_to_pyg(G)
        
        # Get predictions
        embeddings, scores = self.predict(graph_data)
        
        # Calculate centrality measures
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        # Create results DataFrame
        results = []
        for i, company_name in enumerate(graph_data.company_names):
            company_dict = companies_data[i].copy()
            
            # Add GNN scores
            company_dict['gnn_embedding'] = embeddings[i].tolist()
            company_dict['gnn_score'] = float(scores[0])  # Graph-level score
            company_dict['node_centrality'] = centrality.get(company_name, 0)
            company_dict['betweenness_centrality'] = betweenness.get(company_name, 0)
            
            # Combined score
            company_dict['graph_importance'] = (
                0.4 * company_dict['gnn_score'] +
                0.3 * company_dict['node_centrality'] +
                0.3 * company_dict['betweenness_centrality']
            )
            
            results.append(company_dict)
        
        # Create DataFrame and rank
        df = pd.DataFrame(results)
        df = df.sort_values('graph_importance', ascending=False)
        df['gnn_rank'] = range(1, len(df) + 1)
        
        return df
    
    def visualize_graph(self, G: nx.Graph, save_path: Optional[str] = None):
        """Visualize company relationship graph"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Color nodes by sector
        sectors = list(set(nx.get_node_attributes(G, 'sector').values()))
        color_map = {sector: i for i, sector in enumerate(sectors)}
        node_colors = [color_map[G.nodes[node]['sector']] for node in G.nodes()]
        
        # Size nodes by centrality
        centrality = nx.degree_centrality(G)
        node_sizes = [centrality[node] * 3000 for node in G.nodes()]
        
        # Draw graph
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors,
                              node_size=node_sizes,
                              cmap='tab20',
                              alpha=0.8)
        
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        # Add labels for high-centrality nodes
        high_centrality_nodes = {node: node for node, cent in centrality.items() 
                                if cent > np.percentile(list(centrality.values()), 75)}
        nx.draw_networkx_labels(G, pos, high_centrality_nodes, font_size=8)
        
        plt.title("Company Relationship Network", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler': self.scaler
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler = checkpoint['scaler']
        logger.info(f"Model loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Sample company data with ML features
    companies = [
        {
            'company_name': 'TechCorp',
            'sector': 'Technology',
            'sentiment_score': 0.8,
            'acquisition_probability': 0.7,
            'financial_health_score': 0.75,
            'growth_potential': 0.85,
            'market_volatility': 0.4,
            'news_momentum': 0.6,
            'zero_shot_confidence': 0.7
        },
        {
            'company_name': 'FinanceInc',
            'sector': 'Finance',
            'sentiment_score': 0.6,
            'acquisition_probability': 0.5,
            'financial_health_score': 0.8,
            'growth_potential': 0.6,
            'market_volatility': 0.3,
            'news_momentum': 0.4,
            'zero_shot_confidence': 0.5
        }
    ]
    
    # Initialize GNN
    gnn = CompanyGraphNetwork()
    
    # Build and analyze graph
    G = gnn.build_company_graph(companies)
    
    # Get rankings
    rankings = gnn.rank_companies(companies)
    print("\nGNN Company Rankings:")
    print(rankings[['company_name', 'gnn_score', 'graph_importance', 'gnn_rank']])
