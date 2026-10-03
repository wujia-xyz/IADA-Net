# Row aggregation at the encoded-feature interface

The row pool has one query. The `num_heads` argument retained by its constructor does not split that query into multiple heads; the later depth-interaction modules use four heads. A fixed base query produces content-dependent weights through the keys. IADA additionally conditions the query on the global descriptor mean before each row is compressed.

The following analysis treats the encoded descriptors as inputs to aggregation. It does not require the image encoder to realize every theoretical feature grid, and it is not a guarantee of diagnostic or population-risk superiority.

## Lateral permutation invariance

An independent permutation within each row preserves the global mean. It permutes that row's keys, values and weights together, leaving the weighted value sum unchanged. The resulting ordered row sequence is therefore unchanged. Moving descriptors between rows changes which descriptors are compressed together.

## Context before compression

Let an even-sized G-by-G grid contain D-dimensional descriptors with zero channel mean and norm sqrt(D), where D >= 4. Consider a row compressor that applies a continuous scalar score to each descriptor, uses the same scoring/value maps at every lateral position in a row, normalizes scores separately within each row and uses affine values. Its subsequent decoder may use all row summaries and the global descriptor mean.

For every gamma > 0, one fixed image-conditioned affine-query pool with a continuous scalar readout defines a target whose uniform absolute approximation error for every such independent row compressor is at least one half of tanh(gamma)^2.

To see this, fix the adaptive key/value/output maps to identity, its query to gamma G times the global mean divided by sqrt(D), and its readout to the squared norm of the first row divided by D. For a competing first-row scorer, subtract its scores at opposite unit directions in the zero-mean subspace. The difference is continuous and odd. The intermediate value theorem on a semicircle gives a zero v. The same argument on the subspace perpendicular to v gives an orthogonal zero w; D >= 4 ensures that this second semicircle exists.

Put balanced copies of plus/minus sqrt(D)v or plus/minus sqrt(D)w in the first row. Put sqrt(D)v throughout the second row and identical balanced rows elsewhere. Both global means equal sqrt(D)v/G. The independent scorer gives uniform first-row weights for either pair, so affine-value summaries and all decoder inputs coincide. The adaptive outputs are sqrt(D)v tanh(gamma) and zero, producing the stated target gap and half-gap lower bound.

The scoring functions may be nonlinear. The proof still relies on row-local normalization and affine values. It does not cover a global softmax over the entire grid, nonlinear value lifts, position-dependent lateral scorers or a separate CLS-token bypass. This is a one-way separation for the stated representation class, not a strict-superset claim about arbitrary attention networks.

## Local contextual response

For a descriptor in a different row, the adaptive row output has derivative

```text
W_o W_v Sigma_row W_k^T W_a / (G^2 sqrt(D))
```

Here Sigma_row is the attention-weighted covariance of descriptors in the receiving row, and the W matrices are the linear parts of the respective projections. Its rank is at most G - 1 and its cross-row response is zero when W_a is zero. Affine biases cancel in this covariance expression. The statement concerns descriptor perturbations at the aggregation interface; the upstream encoder may itself mix spatial information.

These properties explain the representation's structure. Actual learned behavior and classification performance require their own experimental evidence.
