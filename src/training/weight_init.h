/* weight_init.h -- Depth-muP weight initialization (muPC, arXiv 2505.13124).
 *
 * Depth-muP extends maximal update parameterization (muP) to scale with
 * both width D and depth L. Each layer's contribution to the residual
 * stream is O(1/L), preventing activation growth with depth.
 *
 * Weight init: N(0, (1/sqrt(fan_in)) * (1/sqrt(L)))^2
 * Router: small init for near-uniform routing
 * Embedding: 1/sqrt(D), no depth factor (not in residual stack)
 * RMSNorm gamma: all 1.0 (standard)
 */

#ifndef WEIGHT_INIT_H
#define WEIGHT_INIT_H

#include "hspa_config.h"
#include "hspa_model.h"

/* Initialize all model weights using Depth-muP scheme.
 * Must be called after hspa_model_create and before training. */
void weight_init_depth_mup(HSPAModel *model, const HSPAConfig *cfg);

#endif /* WEIGHT_INIT_H */
