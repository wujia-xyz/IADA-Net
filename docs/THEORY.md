# Context-conditioned row pooling (CCRP)

The manuscript's Proposition 1 analyzes context before row compression. The final URFM-L/16 encoder yields a rectangular notation of R rows and C columns, with R=C=14 and descriptor dimension D=1024 in the implementation.

## Pooling equations and code

Let the encoded patch grid be F=(f[d,i]). CCRP first computes the global mean and conditions its learned query:

```text
g = sum(f[d,i]) / (R C)
q = q0 + Wa g + ba
alpha[d,i] = softmax_i(q^T (WK f[d,i] + bK) / sqrt(D))
x[d] = WO sum_i(alpha[d,i] (WV f[d,i] + bV)) + bO
```

Each row has its own normalization and contributes one descriptor. `iada.layers.ConditionalRowPooling` implements CCRP; `row_pooling.context_proj` implements Wa and ba. The Base model uses the fixed-query pool. The Query model adds the zero-initialized adapter and trains only that adapter, preserving the selected Base output at initialization.

The row pool uses one query; its retained `num_heads` constructor argument does not split it into multiple attention heads. The subsequent DBI layers use four attention heads.

## Proposition 1: context before row compression

Let R >= 2, C be even, D >= 4, r > 0 and gamma > 0. Tokens lie on the zero-channel-mean sphere

```text
S_r = {f in R^D : 1^T f = 0 and ||f|| = r}.
```

A context-free row-local pool scores tokens in row d using a continuous scalar function s[d] shared across lateral positions, applies a row-local softmax and averages affine values A[d]f+c[d]. Scores may be nonlinear and may differ between rows. Its decoder may be any function of every row summary and the global mean.

There is one CCRP pool whose first descriptor defines T(F)=||x[1](F)||²/r², such that every pool in that comparison class and every decoder h satisfy

```text
sup_F |h(row summaries, g) - T(F)| >= 0.5 tanh(gamma)^2.
```

### Witness construction

Choose identity key, value and output maps, zero biases, q0=0 and Wa=(gamma R sqrt(D)/r²)I. For a competing first-row scorer, s[1](r u)-s[1](-r u) is continuous and odd on the unit sphere of the zero-mean subspace. The intermediate value theorem on a great circle gives a zero v. Applying the same argument in the subspace perpendicular to v gives an orthogonal zero w; D >= 4 supplies the required two-dimensional subspace.

In grid A, the first row contains C/2 copies each of r v and -r v. Grid B uses r w and -r w instead. Both grids have r v throughout their second row, with identical balanced rows afterward. Their global means are both r v/R.

The context-free first-row scorer assigns uniform weights in both grids. Its affine summaries, other rows and global mean are identical, so its decoder receives identical inputs. CCRP's conditioned query is gamma sqrt(D) v/r: grid A has first-row scores ±gamma and output r v tanh(gamma), while grid B has zero scores and zero output. The targets differ by tanh(gamma)², and any common decoder output misses one by at least half the gap. Figure 4(b) uses R=4, C=6 and gamma=1.

## Scope and ordered depth readout

The proposition concerns the encoded-feature interface and the stated class of row-local scores and affine values. Global token attention, nonlinear value lifts, position-dependent lateral scores and a CLS-token bypass are outside that comparison class. Classification performance is measured by the manuscript's experiments.

An independent lateral permutation within each row preserves the global mean and the weighted row sums. The row descriptors retain their depth positions. `SimpleDepthEncoding` supplies learned and sinusoidal depth codes; two `BidirectionalInteractionLayer` blocks exchange information between top-down and bottom-up streams. `URFMIADA` realigns and fuses the streams, averages over depth and predicts the malignancy probability.
