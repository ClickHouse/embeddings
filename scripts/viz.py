import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, MDS
import umap

# 26 models in the exact row order of sim_*.csv  (short label, family)
MODELS=[("minilm-l12","ST"),("minilm-l6","ST"),("paraphrase-minilm","ST"),
("bge-base","BGE"),("e5-base","E5"),("mpnet-base","ST"),("multiqa-mpnet","ST"),("gte-base","GTE"),
("bge-large","BGE"),("bge-m3","BGE"),("e5-large","E5"),("mE5-large","E5"),("mistral-embed","Mistral"),
("pplx-0.6b","Perplexity"),("gte-large","GTE"),("codestral","Mistral"),("oai-3-small","OpenAI"),
("ada-002","OpenAI"),("nemotron","NVIDIA"),("pplx-4b","Perplexity"),("qwen3-4b","Qwen"),
("gemini-001","Gemini"),("gemini-2","Gemini"),("gemini-2-prev","Gemini"),("oai-3-large","OpenAI"),("qwen3-8b","Qwen")]
labels=[m[0] for m in MODELS]; fams=[m[1] for m in MODELS]
FAM_ORDER=["ST","BGE","E5","GTE","Mistral","OpenAI","Gemini","Qwen","Perplexity","NVIDIA"]
cmap=plt.get_cmap('tab10'); fam_color={f:cmap(i) for i,f in enumerate(FAM_ORDER)}
colors=[fam_color[f] for f in fams]

CKA=np.loadtxt('sim_CKA.csv',delimiter=',')
D=1.0-CKA; np.fill_diagonal(D,0.0); D=np.clip((D+D.T)/2,0,None)   # symmetric distance

tsne=TSNE(n_components=2,metric='precomputed',init='random',perplexity=6,
          learning_rate='auto',random_state=42).fit_transform(D)
ump=umap.UMAP(n_components=2,metric='precomputed',n_neighbors=8,min_dist=0.35,
              random_state=42).fit_transform(D)
mds_model=MDS(n_components=2,dissimilarity='precomputed',random_state=42,
              n_init=12,max_iter=1000,normalized_stress=True)
mds=mds_model.fit_transform(D)
print(f"MDS normalized stress (Kruskal stress-1) = {mds_model.stress_:.4f}  (lower is better; <0.10 good)")

fig,axes=plt.subplots(1,3,figsize=(26,8.5))
for ax,emb,title in [(axes[0],tsne,"t-SNE"),(axes[1],ump,"UMAP"),
                     (axes[2],mds,f"MDS  (stress={mds_model.stress_:.3f}, distance-faithful)")]:
    ax.scatter(emb[:,0],emb[:,1],c=colors,s=140,edgecolors='black',linewidths=0.6,zorder=3)
    for i,lab in enumerate(labels):
        ax.annotate(lab,(emb[i,0],emb[i,1]),fontsize=8,xytext=(5,3),
                    textcoords='offset points',zorder=4)
    ax.set_title(f"{title}  (26 embedding models, from CKA similarity)",fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor('#cccccc')
handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=fam_color[f],
         markeredgecolor='black',markersize=10,label=f) for f in FAM_ORDER]
fig.legend(handles=handles,loc='lower center',ncol=10,fontsize=10,frameon=False,bbox_to_anchor=(0.5,-0.02))
fig.suptitle("HN-comment embedding models — 2D projection of CKA representational similarity",fontsize=15,y=0.98)
plt.tight_layout(rect=[0,0.03,1,0.96])
plt.savefig('model_similarity_2d.png',dpi=150,bbox_inches='tight')
print("saved model_similarity_2d.png", )
print("families:",{f:fams.count(f) for f in FAM_ORDER})
