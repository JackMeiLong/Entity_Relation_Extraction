# Entity_Relation_Extraction
bert in entity relation extraction  
探索预训练模型在实体关系抽取应用  
假如想法对你有用的话，欢迎Star。  

# Entity Relation Extraction

An Improved Baseline for Sentence-level Relation Extraction

concat

entity_0 + entity_1 + context

```Mermaid
flowchart LR
      A(input)--bert embedding-->B[Bert Input]
      B-->D[Bert Representation]
      D[Bert Representation]-->E[mlp]
      E[mlp]-->F[final output]

```




entity_mask

obj mask

sub mask

obj_rep + sub_rep

```Mermaid
flowchart LR
      A(input)--bert embedding-->B[Bert Input]
      B-->D[Bert Representation]
      D[Bert Representation]-->E[mlp]
      A(input)-- obj entity mask-->C[obj entity idx]
      A(input)-- sub entity mask-->G[sub entity idx]
      C[obj entity idx]-->E[mlp]
      G[sub entity idx]-->E[mlp]
      E[mlp]-->F[final output]

```


entity_type_mask

obj_type mask

sub_type mask

obj_type_rep + sub_type_rep

```Mermaid
flowchart LR
      A(input)--bert embedding-->B[Bert Input]
      B-->D[Bert Representation]
      D[Bert Representation]-->E[mlp]
      A(input)-- obj entity type mask-->C[obj entity type idx]
      A(input)-- sub entity type mask-->G[sub entity type idx]
      C[obj entity type idx]-->E[mlp]
      G[sub entity type idx]-->E[mlp]
      E[mlp]-->F[final output]

```


entity_mask + entity_type

```Mermaid
flowchart LR
      A(input)--bert embedding-->B[Bert Input]
      B-->C[Bert Representation]
      C[Bert Representation]-->D[mlp]
      A(input)-- obj entity type mask-->E[obj entity type idx]
      A(input)-- sub entity type mask-->F[sub entity type idx]
      E[obj entity type idx]-->D[mlp]
      F[sub entity type idx]-->D[mlp]
      D[mlp]-->G[final output]
      A(input)-->H[obj/sub entity type]-->I[mlp]
      I[mlp]-->G[final output]
```






