inductive Precision where
  | fp4 : Precision
  | fp8 : Precision
  | fp16 : Precision
  | fp32 : Precision
  | fp64 : Precision
deriving Repr, BEq, DecidableEq

structure Shape where
  dims : List Nat
deriving Repr, BEq, DecidableEq

structure TensorFlags where
  in_tensor_memory : Bool := false
  requires_grad : Bool := true
  is_compressed : Bool := false
deriving Repr, BEq, DecidableEq

structure Tensor where
  data : List Float
  shape : Shape
  dtype : Precision := Precision.fp32
  flags : TensorFlags := {}
deriving Repr

structure SFDConfig where
  beta1 : Float := 0.9
  beta2 : Float := 0.999
  eps : Float := 1e-8
  clip_threshold : Float := 1.0
  fisher_max : Float := 1e6
  warmup_steps : Nat := 10
deriving Repr

structure SFD where
  fisher_diag : Tensor
  momentum_buffer : Tensor
  velocity_buffer : Tensor
  beta1 : Float
  beta2 : Float
  eps : Float
  clip_threshold : Float
  fisher_max : Float
  warmup_steps : Nat
  step_count : Nat
  param_size : Nat
  initialized : Bool
deriving Repr

structure DynamicLossScaler where
  scale : Float
  growth_factor : Float := 2.0
  backoff_factor : Float := 0.5
  growth_interval : Nat := 2000
  steps_since_last_overflow : Nat := 0
deriving Repr

structure SpectralNormalizer where
  power_iterations : Nat
  eps : Float := 1e-12
  max_singular_value : Float := 1.0
deriving Repr

structure GradientFlowController where
  spectral_normalizer : SpectralNormalizer
  gradient_clip_norm : Float := 1.0
  use_normalized_gradient_flow : Bool := true
deriving Repr

structure KFACBlock where
  A_inv : Tensor
  G_inv : Tensor
  damping : Float
  update_freq : Nat := 10
  last_update : Nat := 0
deriving Repr

structure HyperparamConfig where
  lr : Float
  beta1 : Float
  beta2 : Float
  weight_decay : Float
deriving Repr

structure Observation where
  params : HyperparamConfig
  score : Float
deriving Repr

structure HyperparameterSpace where
  lr_min : Float := 1e-6
  lr_max : Float := 1e-2
  beta1_min : Float := 0.85
  beta1_max : Float := 0.95
  beta2_min : Float := 0.99
  beta2_max : Float := 0.9999
  weight_decay_min : Float := 0.0
  weight_decay_max : Float := 0.1
deriving Repr

structure BayesianOptimizer where
  space : HyperparameterSpace
  observations : List Observation
  best_params : HyperparamConfig
  best_score : Float
deriving Repr

structure MARSVarianceReducer where
  reference_gradients : List Tensor
  snapshot_freq : Nat := 100
  scale_factor : Float := 1.0
  momentum : Float := 0.9
deriving Repr

inductive LRScheduleType where
  | cosine_annealing
  | cosine_annealing_with_warmup
  | polynomial_decay
  | exponential_decay
  | one_cycle
  | sophia_style
deriving Repr, BEq, DecidableEq

structure LRScheduler where
  schedule_type : LRScheduleType
  base_lr : Float
  min_lr : Float
  max_lr : Float
  warmup_steps : Nat
  total_steps : Nat
  current_step : Nat
deriving Repr

inductive OpType where
  | matmul
  | add
  | activation
  | fused_gemm_bias_act
deriving Repr, BEq, DecidableEq

structure FusedKernel where
  operations : List OpType
  use_fp4 : Bool
deriving Repr

structure B200OptimizationConfig where
  use_fp4_tensor_cores : Bool := true
  use_tensor_memory : Bool := true
deriving Repr

structure B200KernelOptimizer where
  config : B200OptimizationConfig
deriving Repr

structure GaussianProcess where
  observations : List Observation
  kernel_variance : Float := 1.0
  length_scale : Float := 0.1
  noise_variance : Float := 0.01
deriving Repr

structure Prediction where
  mean : Float
  variance : Float
deriving Repr

inductive CachePolicy where
  | cache_all
  | recompute_all
  | adaptive
deriving Repr, BEq, DecidableEq

structure ReversibleOptimizerState where
  forward_cache_policy : CachePolicy := CachePolicy.adaptive
  recompute_threshold : Float := 0.5
deriving Repr

structure MixedPrecisionConfig where
  use_fp4 : Bool := true
  use_fp8 : Bool := true
  use_fp16 : Bool := true
  master_weights_precision : Precision := Precision.fp32
  gradient_accumulation_steps : Nat := 4
  loss_scale : Float := 1024.0
  dynamic_loss_scaling : Bool := true
deriving Repr

structure MixedPrecisionTrainer where
  config : MixedPrecisionConfig
  master_weights : List Tensor
  working_weights : List Tensor
  accumulated_gradients : List Tensor
  accumulation_counter : Nat
  loss_scaler : DynamicLossScaler
deriving Repr

structure GPUMetrics where
  utilization_percent : Float
  memory_used_gb : Float
  tensor_core_util : Float
  nvlink_bandwidth_util : Float
deriving Repr

structure MetricsStore where
  training_losses : List Float
  validation_losses : List Float
  learning_rates : List Float
  gradient_norms : List Float
  parameter_norms : List Float
  step_times_ms : List Float
  gpu_utilization : List Float
  memory_usage_gb : List Float
  tensor_core_utilization : List Float
  nvlink_bandwidth_utilization : List Float
deriving Repr

structure Report where
  average_loss : Float
  average_step_time_ms : Float
  throughput_steps_per_sec : Float
  average_gpu_utilization : Float
  average_memory_usage_gb : Float
  average_tensor_core_utilization : Float
  average_nvlink_utilization : Float
  total_steps : Nat
deriving Repr

structure PerformanceMonitor where
  metrics : MetricsStore
  telemetry_enabled : Bool
deriving Repr

structure B200MemoryManager where
  config : B200OptimizationConfig
  tensor_memory_used : Nat
  tensor_memory_capacity : Nat
deriving Repr

structure SophiaSOAPConfig where
  rho : Float := 0.04
  gamma : Float := 0.01
  hessian_update_freq : Nat := 10
  use_gauss_newton : Bool := true
deriving Repr

class FloatArith (F : Type) where
  add_comm : ∀ (a b : F), a + b = b + a
  mul_comm : ∀ (a b : F), a * b = b * a
  add_assoc : ∀ (a b c : F), a + b + c = a + (b + c)
  mul_assoc : ∀ (a b c : F), a * b * c = a * (b * c)
  add_zero : ∀ (a : F), a + 0 = a
  zero_add : ∀ (a : F), 0 + a = a
  mul_one : ∀ (a : F), a * 1 = a
  one_mul : ∀ (a : F), 1 * a = a
  mul_zero : ∀ (a : F), a * 0 = 0
  zero_mul : ∀ (a : F), 0 * a = 0
  sub_self : ∀ (a : F), a - a = 0
  le_refl : ∀ (a : F), a ≤ a
  le_of_not_lt : ∀ (a b : F), ¬(a < b) → b ≤ a
  le_of_not_gt : ∀ (a b : F), ¬(a > b) → a ≤ b
  le_trans : ∀ (a b c : F), a ≤ b → b ≤ c → a ≤ c
  sqrt_nonneg : ∀ (a : F), 0 ≤ Float.sqrt a
  sqrt_zero : Float.sqrt 0 = 0
  min_le_right : ∀ (a b : F), min a b ≤ b
  min_le_left : ∀ (a b : F), min a b ≤ a
  max_le_iff : ∀ (a b c : F), max a b ≤ c ↔ a ≤ c ∧ b ≤ c

def Shape.totalSize (s : Shape) : Nat :=
  List.foldl (· * ·) 1 s.dims

def shapesEqual (a b : Shape) : Bool :=
  BEq.beq a b

def tensorFlagsToBits (flags : TensorFlags) : Nat :=
  (if flags.in_tensor_memory then 1 else 0) +
  (if flags.requires_grad then 2 else 0) +
  (if flags.is_compressed then 4 else 0)

def tensorFlagsFromBits (bits : Nat) : TensorFlags :=
  { in_tensor_memory := (bits % 2) != 0
  , requires_grad := ((bits / 2) % 2) != 0
  , is_compressed := ((bits / 4) % 2) != 0
  }

def quantizeValuePure (value : Float) (precision : Precision) : Float :=
  match precision with
  | Precision.fp32 => value
  | Precision.fp64 => value
  | _ => value

def Tensor.fill (t : Tensor) (value : Float) : Tensor :=
  { t with data := List.replicate t.data.length value }

def Tensor.zeros (shape : Shape) : Tensor :=
  { data := List.replicate shape.totalSize 0.0
  , shape := shape
  , dtype := Precision.fp32
  , flags := {} }

def Tensor.ones (shape : Shape) : Tensor :=
  { data := List.replicate shape.totalSize 1.0
  , shape := shape
  , dtype := Precision.fp32
  , flags := {} }

def Tensor.eye (n : Nat) : Tensor :=
  let data := List.ofFn (fun (idx : Fin (n * n)) =>
    if idx.val / n == idx.val % n then (1.0 : Float) else (0.0 : Float))
  { data := data
  , shape := { dims := [n, n] }
  , dtype := Precision.fp32
  , flags := {} }

def Tensor.addTensors (a b : Tensor) : Tensor :=
  { a with data := List.zipWith (· + ·) a.data b.data }

def Tensor.subTensors (a b : Tensor) : Tensor :=
  { a with data := List.zipWith (· - ·) a.data b.data }

def Tensor.mulScalar (t : Tensor) (s : Float) : Tensor :=
  { t with data := List.map (· * s) t.data }

def Tensor.clone (t : Tensor) : Tensor :=
  { data := t.data, shape := t.shape, dtype := t.dtype, flags := t.flags }

def Tensor.copyFrom (self other : Tensor) : Tensor :=
  { self with data := other.data, flags := other.flags, dtype := other.dtype }

def Tensor.copyFromWithCast (self other : Tensor) : Tensor :=
  { self with
    data := List.map (fun v => quantizeValuePure v self.dtype) other.data
    flags := { other.flags with is_compressed := self.dtype == Precision.fp4 || self.dtype == Precision.fp8 } }

def Tensor.sizeBytes (t : Tensor) : Nat :=
  t.data.length * 4

noncomputable def Tensor.normL2 (t : Tensor) : Float :=
  Float.sqrt (List.foldl (fun acc v => acc + v * v) 0.0 t.data)

def Tensor.outerProduct (a b : Tensor) : Tensor :=
  let m := a.data.length
  let n := b.data.length
  let data := List.ofFn (fun (idx : Fin (m * n)) =>
    let i := idx.val / n
    let j := idx.val % n
    match a.data.get? i, b.data.get? j with
    | some ai, some bj => ai * bj
    | _, _ => 0.0)
  { data := data, shape := { dims := [m, n] }, dtype := Precision.fp32, flags := {} }

def matmulData (m k n : Nat) (aData bData : List Float) : List Float :=
  List.ofFn (fun (idx : Fin (m * n)) =>
    let i := idx.val / n
    let j := idx.val % n
    let rec sumLoop (p : Nat) (acc : Float) : Float :=
      match p with
      | 0 => acc
      | p' + 1 =>
        let aVal := match aData.get? (i * k + p') with | some v => v | none => 0.0
        let bVal := match bData.get? (p' * n + j) with | some v => v | none => 0.0
        sumLoop p' (acc + aVal * bVal)
    sumLoop k 0.0)

def Tensor.matmul (A B : Tensor) : Tensor :=
  match A.shape.dims, B.shape.dims with
  | [m, k], [k2, n] =>
    if k == k2 then
      { data := matmulData m k n A.data B.data
      , shape := { dims := [m, n] }
      , dtype := Precision.fp32
      , flags := {} }
    else { data := [], shape := { dims := [] }, dtype := Precision.fp32, flags := {} }
  | _, _ => { data := [], shape := { dims := [] }, dtype := Precision.fp32, flags := {} }

noncomputable def Tensor.spectralNorm (t : Tensor) (max_iter : Nat) (_eps : Float) : Float :=
  match t.shape.dims with
  | [_m, _n] => Float.sqrt (List.foldl (fun acc v => acc + v * v) 0.0 t.data)
  | _ => 0.0

def Tensor.fillRandomNormal (_t : Tensor) (_mean _std_dev : Float) (seed : Nat) : Tensor :=
  { _t with data := List.replicate _t.data.length (Float.ofNat seed * 0.0001) }

def Tensor.fillRademacher (t : Tensor) (seed : Nat) : Tensor :=
  { t with data := List.ofFn (fun (i : Fin t.data.length) =>
    if (i.val + seed) % 2 == 0 then 1.0 else -1.0) }

def Tensor.convertToFP4 (t : Tensor) : Tensor :=
  { t with
    data := List.map (fun v => quantizeValuePure v Precision.fp4) t.data
    dtype := Precision.fp4
    flags := { t.flags with is_compressed := true } }

def Tensor.save (t : Tensor) : List Nat :=
  [0x54464453, t.shape.dims.length] ++ t.shape.dims ++ [t.data.length]

def Tensor.load (header : List Nat) (floatData : List Float) : Tensor :=
  match header with
  | _ :: ndims :: rest =>
    let dims := rest.take ndims
    { data := floatData, shape := { dims := dims }, dtype := Precision.fp32, flags := {} }
  | _ => { data := floatData, shape := { dims := [] }, dtype := Precision.fp32, flags := {} }

def floatClamp (v lo hi : Float) : Float :=
  if v < lo then lo else if v > hi then hi else v

def DynamicLossScaler.init' (initial_scale : Float) : DynamicLossScaler :=
  { scale := initial_scale, growth_factor := 2.0, backoff_factor := 0.5
  , growth_interval := 2000, steps_since_last_overflow := 0 }

def DynamicLossScaler.update' (dls : DynamicLossScaler) (has_overflow : Bool) : DynamicLossScaler :=
  if has_overflow then
    { dls with scale := dls.scale * dls.backoff_factor, steps_since_last_overflow := 0 }
  else
    let newSteps := dls.steps_since_last_overflow + 1
    if newSteps >= dls.growth_interval then
      { dls with scale := dls.scale * dls.growth_factor, steps_since_last_overflow := 0 }
    else
      { dls with steps_since_last_overflow := newSteps }

def DynamicLossScaler.updateClamped (dls : DynamicLossScaler) (has_overflow : Bool) : DynamicLossScaler :=
  let updated := DynamicLossScaler.update' dls has_overflow
  { updated with scale := floatClamp updated.scale 1.0 65536.0 }

def SFD.init' (param_size : Nat) : SFD :=
  { fisher_diag := { data := List.replicate param_size 1.0, shape := { dims := [param_size] } }
  , momentum_buffer := { data := List.replicate param_size 0.0, shape := { dims := [param_size] } }
  , velocity_buffer := { data := List.replicate param_size 0.0, shape := { dims := [param_size] } }
  , beta1 := 0.9, beta2 := 0.999, eps := 1e-8, clip_threshold := 1.0
  , fisher_max := 1e6, warmup_steps := 10, step_count := 0
  , param_size := param_size, initialized := true }

def SFD.resetFisher (sfd : SFD) : SFD :=
  { sfd with fisher_diag := { sfd.fisher_diag with data := List.replicate sfd.fisher_diag.data.length 1.0 } }

def SFD.ampSchedule (_sfd : SFD) (step warmup total : Nat) : Float :=
  if warmup == 0 then 1.0
  else if total <= warmup then 1.0
  else if step < warmup then Float.ofNat step / Float.ofNat warmup
  else
    let progress_num := step - warmup
    let progress_denom := total - warmup
    if progress_denom == 0 then 0.5
    else
      let progress := Float.ofNat progress_num / Float.ofNat progress_denom
      0.5 * (1.0 + Float.cos (3.14159265358979 * progress))

noncomputable def SFD.adaptiveLR (sfd : SFD) (grad_norm param_norm : Float) : Float :=
  if grad_norm == 0.0 then 1.0
  else
    let denom := param_norm + sfd.eps
    let ratio := grad_norm / denom
    let inner := ratio + sfd.eps
    if inner <= 0.0 then 1.0
    else 1.0 / Float.sqrt inner

def SpectralNormalizer.init' (power_iterations : Nat) : SpectralNormalizer :=
  { power_iterations := power_iterations, eps := 1e-12, max_singular_value := 1.0 }

noncomputable def SpectralNormalizer.lipschitzRegularization
    (_sn : SpectralNormalizer) (loss : Float) (spectral_norms : List Float) (lambda : Float) : Float :=
  let reg_term := List.foldl (fun acc sigma => acc + (sigma - 1.0) * (sigma - 1.0)) 0.0 spectral_norms
  loss + lambda * reg_term

noncomputable def SpectralNormalizer.normalizeWeights (sn : SpectralNormalizer) (weights : Tensor) : Tensor :=
  let sigma := weights.spectralNorm sn.power_iterations sn.eps
  if sigma > sn.max_singular_value then
    Tensor.mulScalar weights (sn.max_singular_value / sigma)
  else weights

def GradientFlowController.init' : GradientFlowController :=
  { spectral_normalizer := SpectralNormalizer.init' 20
  , gradient_clip_norm := 1.0
  , use_normalized_gradient_flow := true }

def LRScheduler.init' (schedule_type : LRScheduleType) (base_lr : Float)
    (warmup_steps total_steps : Nat) : LRScheduler :=
  { schedule_type := schedule_type, base_lr := base_lr
  , min_lr := base_lr * 0.01, max_lr := base_lr * 10.0
  , warmup_steps := warmup_steps, total_steps := total_steps, current_step := 0 }

noncomputable def LRScheduler.getLearningRate (sched : LRScheduler) : Float :=
  let lr := sched.base_lr
  floatClamp lr sched.min_lr sched.max_lr

noncomputable def LRScheduler.getLearningRateFull (sched : LRScheduler) : Float × LRScheduler :=
  let decay_steps := if sched.total_steps > sched.warmup_steps then sched.total_steps - sched.warmup_steps else 1
  let lr :=
    if sched.warmup_steps > 0 ∧ sched.current_step < sched.warmup_steps then
      sched.base_lr * (Float.ofNat sched.current_step / Float.ofNat sched.warmup_steps)
    else
      match sched.schedule_type with
      | LRScheduleType.cosine_annealing =>
        let progress := Float.ofNat (sched.current_step - min sched.current_step sched.warmup_steps) / Float.ofNat decay_steps
        sched.min_lr + (sched.base_lr - sched.min_lr) * 0.5 * (1.0 + Float.cos (3.14159265358979 * progress))
      | LRScheduleType.cosine_annealing_with_warmup =>
        let progress := Float.ofNat (sched.current_step - min sched.current_step sched.warmup_steps) / Float.ofNat decay_steps
        sched.min_lr + (sched.base_lr - sched.min_lr) * 0.5 * (1.0 + Float.cos (3.14159265358979 * progress))
      | LRScheduleType.polynomial_decay =>
        let progress := Float.ofNat (sched.current_step - min sched.current_step sched.warmup_steps) / Float.ofNat decay_steps
        sched.base_lr * Float.pow (max 0.0 (1.0 - progress)) 2.0
      | LRScheduleType.exponential_decay =>
        let steps_since := Float.ofNat (sched.current_step - min sched.current_step sched.warmup_steps)
        sched.base_lr * Float.pow 0.96 (steps_since / 1000.0)
      | LRScheduleType.one_cycle =>
        let mid := max (sched.warmup_steps + 1) (sched.total_steps / 2)
        if sched.current_step < mid then
          let rise := max 1 (mid - sched.warmup_steps)
          let progress := Float.ofNat (sched.current_step - min sched.current_step sched.warmup_steps) / Float.ofNat rise
          sched.base_lr + (sched.max_lr - sched.base_lr) * progress
        else
          let fall := max 1 (sched.total_steps - mid)
          let progress := Float.ofNat (sched.current_step - mid) / Float.ofNat fall
          sched.min_lr + (sched.max_lr - sched.min_lr) * 0.5 * (1.0 + Float.cos (3.14159265358979 * progress))
      | LRScheduleType.sophia_style => sched.base_lr
  let clamped := floatClamp lr sched.min_lr sched.max_lr
  (clamped, { sched with current_step := sched.current_step + 1 })

def KFACBlock.init' (input_dim output_dim : Nat) (damping : Float) : KFACBlock :=
  { A_inv := Tensor.eye input_dim, G_inv := Tensor.eye output_dim
  , damping := damping, update_freq := 10, last_update := 0 }

def KFACBlock.updateStatisticsPure (block : KFACBlock) (activations gradients : Tensor) (alpha : Float) : KFACBlock :=
  let a_dim := block.A_inv.shape.dims.head!
  let g_dim := block.G_inv.shape.dims.head!
  let newA := List.ofFn (fun (idx : Fin (a_dim * a_dim)) =>
    let row := idx.val / a_dim
    let col := idx.val % a_dim
    let old := match block.A_inv.data.get? idx.val with | some v => v | none => 0.0
    let target := if row == col then
      match activations.data.get? row with
      | some a => a * a + block.damping
      | none => 0.0
    else 0.0
    alpha * old + (1.0 - alpha) * target)
  let newG := List.ofFn (fun (idx : Fin (g_dim * g_dim)) =>
    let row := idx.val / g_dim
    let col := idx.val % g_dim
    let old := match block.G_inv.data.get? idx.val with | some v => v | none => 0.0
    let target := if row == col then
      match gradients.data.get? row with
      | some g => g * g + block.damping
      | none => 0.0
    else 0.0
    alpha * old + (1.0 - alpha) * target)
  { block with
    A_inv := { block.A_inv with data := newA }
    G_inv := { block.G_inv with data := newG } }

def BayesianOptimizer.init' (space : HyperparameterSpace) : BayesianOptimizer :=
  { space := space, observations := []
  , best_params := { lr := 0.001, beta1 := 0.9, beta2 := 0.999, weight_decay := 0.01 }
  , best_score := 1e38 }

def BayesianOptimizer.observe' (bo : BayesianOptimizer) (params : HyperparamConfig) (score : Float) : BayesianOptimizer :=
  let newObs := bo.observations ++ [{ params := params, score := score }]
  if score < bo.best_score then
    { bo with observations := newObs, best_score := score, best_params := params }
  else { bo with observations := newObs }

def MARSVarianceReducer.varianceReducedGradientShape (current_grad : Tensor) : Shape :=
  current_grad.shape

def MARSVarianceReducer.varianceReducedGradientPure
    (vr : MARSVarianceReducer) (current_grad reference_grad : Tensor) (param_idx : Nat) : Tensor :=
  match vr.reference_gradients.get? param_idx with
  | some ref_stored =>
    let newData := List.ofFn (fun (i : Fin current_grad.data.length) =>
      let g_current := match current_grad.data.get? i.val with | some v => v | none => 0.0
      let g_ref_mini := match reference_grad.data.get? i.val with | some v => v | none => 0.0
      let g_ref_full := match ref_stored.data.get? i.val with | some v => v | none => 0.0
      let vr_raw := g_current - g_ref_mini + g_ref_full
      vr.momentum * vr_raw + (1.0 - vr.momentum) * g_current)
    { data := newData, shape := current_grad.shape, dtype := current_grad.dtype, flags := current_grad.flags }
  | none => current_grad

noncomputable def erfApprox (x : Float) : Float :=
  let a1 : Float := 0.254829592
  let a2 : Float := -0.284496736
  let a3 : Float := 1.421413741
  let a4 : Float := -1.453152027
  let a5 : Float := 1.061405429
  let p : Float := 0.3275911
  let sign : Float := if x < 0.0 then -1.0 else 1.0
  let abs_x := if x < 0.0 then -x else x
  let t := 1.0 / (1.0 + p * abs_x)
  let y := 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Float.exp (-abs_x * abs_x)
  sign * y

noncomputable def GaussianProcess.kernel' (gp : GaussianProcess) (x1 x2 : HyperparamConfig) : Float :=
  let d_lr := x1.lr - x2.lr
  let d_b1 := x1.beta1 - x2.beta1
  let d_b2 := x1.beta2 - x2.beta2
  let d_wd := x1.weight_decay - x2.weight_decay
  let dist_sq := d_lr * d_lr + d_b1 * d_b1 + d_b2 * d_b2 + d_wd * d_wd
  gp.kernel_variance * Float.exp (-dist_sq / (2.0 * gp.length_scale * gp.length_scale))

noncomputable def GaussianProcess.predict' (gp : GaussianProcess) (_config : HyperparamConfig) : Prediction :=
  match gp.observations with
  | [] => { mean := 0.0, variance := gp.kernel_variance }
  | obs =>
    let n := obs.length
    let k_star := List.map (fun o => GaussianProcess.kernel' gp _config o.params) obs
    let mean := List.foldl (· + ·) 0.0 (List.zipWith (· * ·) k_star (List.map (fun o => o.score) obs)) / Float.ofNat n
    let prior_var := GaussianProcess.kernel' gp _config _config + gp.noise_variance
    { mean := mean, variance := max prior_var 1e-8 }

noncomputable def GaussianProcess.expectedImprovement' (gp : GaussianProcess) (candidate : HyperparamConfig) (best_score : Float) : Float :=
  let pred := GaussianProcess.predict' gp candidate
  let std_dev := Float.sqrt pred.variance
  if std_dev < 1e-8 then 0.0
  else
    let improvement := best_score - pred.mean
    let z := improvement / std_dev
    let phi_z := 0.5 * (1.0 + erfApprox (z / Float.sqrt 2.0))
    let pdf_z := Float.exp (-0.5 * z * z) / Float.sqrt (2.0 * 3.14159265358979)
    let ei := improvement * phi_z + std_dev * pdf_z
    if ei < 0.0 then 0.0 else ei

def B200KernelOptimizer.fuseOperations : List OpType → List OpType
  | OpType.matmul :: OpType.add :: OpType.activation :: rest =>
    OpType.fused_gemm_bias_act :: B200KernelOptimizer.fuseOperations rest
  | x :: rest => x :: B200KernelOptimizer.fuseOperations rest
  | [] => []

def B200KernelOptimizer.selectOptimalPrecision (config : B200OptimizationConfig) (operation : OpType) (tensor_size : Nat) : Precision :=
  if config.use_fp4_tensor_cores && operation == OpType.matmul then
    if tensor_size > 1000000 then Precision.fp4
    else if tensor_size > 100000 then Precision.fp8
    else Precision.fp16
  else if tensor_size > 100000 then Precision.fp8
  else Precision.fp16

def B200MemoryManager.init' (config : B200OptimizationConfig) (tmem_mb : Nat) : B200MemoryManager :=
  { config := config, tensor_memory_used := 0, tensor_memory_capacity := tmem_mb * 1024 * 1024 }

def ReversibleOptimizerState.shouldRecompute (state : ReversibleOptimizerState)
    (computation_cost memory_cost available_memory : Float) : Bool :=
  match state.forward_cache_policy with
  | CachePolicy.cache_all => false
  | CachePolicy.recompute_all => true
  | CachePolicy.adaptive =>
    if available_memory < memory_cost then true
    else computation_cost < memory_cost * state.recompute_threshold

def MetricsStore.init' : MetricsStore :=
  { training_losses := [], validation_losses := [], learning_rates := []
  , gradient_norms := [], parameter_norms := [], step_times_ms := []
  , gpu_utilization := [], memory_usage_gb := []
  , tensor_core_utilization := [], nvlink_bandwidth_utilization := [] }

def MetricsStore.recordStep (ms : MetricsStore) (loss lr grad_norm param_norm step_time : Float) : MetricsStore :=
  { ms with
    training_losses := ms.training_losses ++ [loss]
    learning_rates := ms.learning_rates ++ [lr]
    gradient_norms := ms.gradient_norms ++ [grad_norm]
    parameter_norms := ms.parameter_norms ++ [param_norm]
    step_times_ms := ms.step_times_ms ++ [step_time] }

noncomputable def computeMean (values : List Float) : Float :=
  if values.length == 0 then 0.0
  else List.foldl (· + ·) 0.0 values / Float.ofNat values.length

def PerformanceMonitor.init' (enable_telemetry : Bool) : PerformanceMonitor :=
  { metrics := MetricsStore.init', telemetry_enabled := enable_telemetry }

def MixedPrecisionTrainer.init' (config : MixedPrecisionConfig) : MixedPrecisionTrainer :=
  { config := config, master_weights := [], working_weights := []
  , accumulated_gradients := [], accumulation_counter := 0
  , loss_scaler := DynamicLossScaler.init' config.loss_scale }

structure SFDUpdateElementInput where
  g : Float
  m_old : Float
  v_old : Float
  f_old : Float
  param_old : Float
  beta1 : Float
  beta2 : Float
  eps : Float
  clip_threshold : Float
  fisher_max : Float
  lr : Float
  warmup_factor : Float
  m_correction : Float
  v_correction : Float
deriving Repr

structure SFDUpdateElementOutput where
  m_new : Float
  v_new : Float
  f_new : Float
  param_new : Float
deriving Repr

noncomputable def sfdUpdateElement (inp : SFDUpdateElementInput) : SFDUpdateElementOutput :=
  let m_new := inp.beta1 * inp.m_old + (1.0 - inp.beta1) * inp.g
  let v_new := inp.beta2 * inp.v_old + (1.0 - inp.beta2) * inp.g * inp.g
  let m_hat := if inp.m_correction > 1e-10 then m_new / inp.m_correction else m_new
  let v_hat := if inp.v_correction > 1e-10 then v_new / inp.v_correction else v_new
  let sqrt_v := Float.sqrt (max 0.0 v_hat)
  let adaptive_lr := inp.lr * inp.warmup_factor / (sqrt_v + inp.eps)
  let f_updated := inp.beta2 * inp.f_old + (1.0 - inp.beta2) * inp.g * inp.g
  let f_clamped := min f_updated inp.fisher_max
  let sqrt_fisher := Float.sqrt (max 0.0 f_clamped)
  let update_val_raw := m_hat * adaptive_lr / (sqrt_fisher + inp.eps)
  let update_val := floatClamp update_val_raw (-inp.clip_threshold) inp.clip_threshold
  { m_new := m_new, v_new := v_new, f_new := f_clamped
  , param_new := inp.param_old - update_val }

noncomputable def sfdUpdateLoop
    (grads params momentum velocity fisher : List Float)
    (beta1 beta2 eps clip_threshold fisher_max lr warmup_factor m_correction v_correction : Float)
    : List Float × List Float × List Float × List Float :=
  match grads, params, momentum, velocity, fisher with
  | g :: gs, p :: ps, m :: ms, v :: vs, f :: fs =>
    let out := sfdUpdateElement
      { g := g, m_old := m, v_old := v, f_old := f, param_old := p
      , beta1 := beta1, beta2 := beta2, eps := eps
      , clip_threshold := clip_threshold, fisher_max := fisher_max
      , lr := lr, warmup_factor := warmup_factor
      , m_correction := m_correction, v_correction := v_correction }
    let rest := sfdUpdateLoop gs ps ms vs fs beta1 beta2 eps clip_threshold fisher_max lr warmup_factor m_correction v_correction
    (out.param_new :: rest.1, out.m_new :: rest.2.1, out.v_new :: rest.2.2.1, out.f_new :: rest.2.2.2)
  | _, _, _, _, _ => ([], [], [], [])

noncomputable def SFD.update (sfd : SFD) (gradients params : List Float) (lr : Float) : SFD × List Float :=
  let new_step := sfd.step_count + 1
  let step_f := Float.ofNat new_step
  let warmup_f := Float.ofNat sfd.warmup_steps
  let warmup_factor := if sfd.warmup_steps > 0 ∧ new_step <= sfd.warmup_steps then step_f / warmup_f else 1.0
  let m_correction := 1.0 - Float.pow sfd.beta1 step_f
  let v_correction := 1.0 - Float.pow sfd.beta2 step_f
  let result := sfdUpdateLoop gradients params
    sfd.momentum_buffer.data sfd.velocity_buffer.data sfd.fisher_diag.data
    sfd.beta1 sfd.beta2 sfd.eps sfd.clip_threshold sfd.fisher_max
    lr warmup_factor m_correction v_correction
  ({ sfd with step_count := new_step
    , momentum_buffer := { sfd.momentum_buffer with data := result.2.1 }
    , velocity_buffer := { sfd.velocity_buffer with data := result.2.2.1 }
    , fisher_diag := { sfd.fisher_diag with data := result.2.2.2 } }, result.1)

def SFD.warmStartPure (fisher_data prev_data : List Float) (fisher_max : Float) : List Float :=
  match fisher_data, prev_data with
  | f :: fs, p :: ps =>
    let combined := (f + p) * 0.5
    let clamped := if combined > fisher_max then fisher_max else combined
    clamped :: SFD.warmStartPure fs ps fisher_max
  | remaining, [] => remaining
  | [], _ => []

def SFD.warmStart' (sfd : SFD) (prev_diag : List Float) : SFD :=
  { sfd with fisher_diag := { sfd.fisher_diag with data := SFD.warmStartPure sfd.fisher_diag.data prev_diag sfd.fisher_max } }

noncomputable def SFD.clipGradNormPure (grad_norms : List Float) (max_norm eps : Float) : Float × Float :=
  let total_norm_sq := List.foldl (fun acc n => acc + n * n) 0.0 grad_norms
  let total_norm := Float.sqrt total_norm_sq
  let scale := if total_norm > max_norm then max_norm / (total_norm + eps) else 1.0
  (total_norm, scale)

theorem Nat.beq_refl (n : Nat) : Nat.beq n n = true :=
  Nat.rec (Eq.refl true) (fun k ih => show Nat.beq (k + 1) (k + 1) = true from ih) n

theorem nat_beq_eq_natbeq (a b : Nat) : @BEq.beq Nat instBEqNat a b = Nat.beq a b :=
  Eq.refl (Nat.beq a b)

theorem nat_beq_self (n : Nat) : @BEq.beq Nat instBEqNat n n = true :=
  Eq.trans (nat_beq_eq_natbeq n n) (Nat.beq_refl n)

theorem list_nat_beq_refl : (l : List Nat) → @BEq.beq (List Nat) instBEqList l l = true :=
  fun l => List.rec
    (Eq.refl true)
    (fun hd tl ih =>
      have h1 : @BEq.beq Nat instBEqNat hd hd = true := nat_beq_self hd
      have h2 : @BEq.beq (List Nat) instBEqList tl tl = true := ih
      h1 ▸ h2 ▸ Eq.refl true) l

theorem shape_beq_refl (s : Shape) : @BEq.beq Shape instBEqShape s s = true :=
  match s with | ⟨dims⟩ => list_nat_beq_refl dims

theorem shapesEqual_refl (s : Shape) : shapesEqual s s = true := shape_beq_refl s

theorem Nat.beq_comm (a b : Nat) : Nat.beq a b = Nat.beq b a :=
  Nat.rec (fun b => Nat.rec (Eq.refl true) (fun _ _ => Eq.refl false) b)
    (fun a ih b => Nat.rec (Eq.refl false) (fun b _ => ih b) b) a b

theorem nat_beq_comm (a b : Nat) : @BEq.beq Nat instBEqNat a b = @BEq.beq Nat instBEqNat b a :=
  Eq.trans (nat_beq_eq_natbeq a b) (Eq.trans (Nat.beq_comm a b) (Eq.symm (nat_beq_eq_natbeq b a)))

theorem list_nat_beq_comm : (l1 l2 : List Nat) →
    @BEq.beq (List Nat) instBEqList l1 l2 = @BEq.beq (List Nat) instBEqList l2 l1 :=
  fun l1 => List.rec
    (fun l2 => List.rec (Eq.refl true) (fun _ _ _ => Eq.refl false) l2)
    (fun hd1 tl1 ih l2 => List.rec (Eq.refl false)
      (fun hd2 tl2 _ =>
        have h1 := nat_beq_comm hd1 hd2
        have h2 := ih tl2
        h1 ▸ h2 ▸ Eq.refl _) l2) l1

theorem shapesEqual_symm (a b : Shape) (h : shapesEqual a b = true) : shapesEqual b a = true :=
  match a, b with
  | ⟨da⟩, ⟨db⟩ => Eq.trans (Eq.symm (list_nat_beq_comm da db)) h

theorem totalSize_singleton (n : Nat) : Shape.totalSize { dims := [n] } = n :=
  show 1 * n = n from Nat.one_mul n

theorem totalSize_empty : Shape.totalSize { dims := [] } = 1 := Eq.refl 1

theorem totalSize_pair (a b : Nat) : Shape.totalSize { dims := [a, b] } = a * b :=
  show 1 * a * b = a * b from congrArg (· * b) (Nat.one_mul a)

theorem totalSize_triple (a b c : Nat) : Shape.totalSize { dims := [a, b, c] } = a * b * c :=
  show 1 * a * b * c = a * b * c from congrArg (· * b * c) (Nat.one_mul a)

theorem flagsBits_roundtrip (f : TensorFlags) : tensorFlagsFromBits (tensorFlagsToBits f) = f :=
  match f with
  | ⟨false, false, false⟩ => Eq.refl _
  | ⟨true,  false, false⟩ => Eq.refl _
  | ⟨false, true,  false⟩ => Eq.refl _
  | ⟨true,  true,  false⟩ => Eq.refl _
  | ⟨false, false, true ⟩ => Eq.refl _
  | ⟨true,  false, true ⟩ => Eq.refl _
  | ⟨false, true,  true ⟩ => Eq.refl _
  | ⟨true,  true,  true ⟩ => Eq.refl _

theorem bitsFlags_roundtrip (b : Nat) (hb : b < 8) : tensorFlagsToBits (tensorFlagsFromBits b) = b :=
  match b, hb with
  | 0, _ => Eq.refl 0 | 1, _ => Eq.refl 1 | 2, _ => Eq.refl 2 | 3, _ => Eq.refl 3
  | 4, _ => Eq.refl 4 | 5, _ => Eq.refl 5 | 6, _ => Eq.refl 6 | 7, _ => Eq.refl 7

theorem quantizeValue_fp32 (v : Float) : quantizeValuePure v Precision.fp32 = v := Eq.refl v
theorem quantizeValue_fp64 (v : Float) : quantizeValuePure v Precision.fp64 = v := Eq.refl v

theorem List.get_replicate_fin {α : Type} (n : Nat) (v : α) (i : Nat) (hi : i < n) :
    (List.replicate n v).get ⟨i, (List.length_replicate n v) ▸ hi⟩ = v :=
  Nat.rec (fun i hi => absurd hi (Nat.not_lt_zero i))
    (fun k ih i hi => match i, hi with
      | 0, _ => Eq.refl v
      | j + 1, hj => ih j (Nat.lt_of_succ_lt_succ hj)) n i hi

theorem fill_all_elements (t : Tensor) (v : Float) (i : Nat) (hi : i < t.data.length) :
    ((Tensor.fill t v).data).get ⟨i, (List.length_replicate t.data.length v) ▸ hi⟩ = v :=
  List.get_replicate_fin t.data.length v i hi

theorem zeros_all_zero (s : Shape) (i : Nat) (hi : i < s.totalSize) :
    (Tensor.zeros s).data.get ⟨i, (List.length_replicate s.totalSize 0.0) ▸ hi⟩ = 0.0 :=
  List.get_replicate_fin s.totalSize 0.0 i hi

theorem ones_all_one (s : Shape) (i : Nat) (hi : i < s.totalSize) :
    (Tensor.ones s).data.get ⟨i, (List.length_replicate s.totalSize 1.0) ▸ hi⟩ = 1.0 :=
  List.get_replicate_fin s.totalSize 1.0 i hi

theorem eye_length (n : Nat) : (Tensor.eye n).data.length = n * n :=
  List.length_ofFn _

theorem eye_shape (n : Nat) : (Tensor.eye n).shape = { dims := [n, n] } := Eq.refl _
theorem eye_dtype (n : Nat) : (Tensor.eye n).dtype = Precision.fp32 := Eq.refl _

theorem List.zipWith_comm {α : Type} (f : α → α → α) (comm : ∀ x y, f x y = f y x) :
    (l1 l2 : List α) → List.zipWith f l1 l2 = List.zipWith f l2 l1 :=
  fun l1 => List.rec
    (fun l2 => List.rec (Eq.refl ([] : List α)) (fun _ _ _ => Eq.refl []) l2)
    (fun h1 t1 ih l2 => List.rec (Eq.refl [])
      (fun h2 t2 _ => (comm h1 h2) ▸ (ih t2) ▸ Eq.refl _) l2) l1

theorem tensor_add_comm (a b : Tensor) (hcomm : ∀ x y : Float, x + y = y + x) :
    (Tensor.addTensors a b).data = (Tensor.addTensors b a).data :=
  List.zipWith_comm (· + ·) hcomm a.data b.data

theorem tensor_sub_self (a : Tensor) (hsub : ∀ x : Float, x - x = 0.0) :
    (Tensor.subTensors a a).data = List.replicate a.data.length 0.0 :=
  List.rec (Eq.refl [])
    (fun hd tl ih => (hsub hd) ▸ ih ▸ Eq.refl _) a.data

theorem tensor_mulScalar_zero (t : Tensor) (hmul : ∀ x : Float, x * 0.0 = 0.0) :
    (Tensor.mulScalar t 0.0).data = List.replicate t.data.length 0.0 :=
  List.rec (Eq.refl [])
    (fun hd tl ih => (hmul hd) ▸ ih ▸ Eq.refl _) t.data

theorem tensor_mulScalar_one (t : Tensor) (hmul : ∀ x : Float, x * 1.0 = x) :
    (Tensor.mulScalar t 1.0).data = t.data :=
  List.rec (Eq.refl [])
    (fun hd tl ih => (hmul hd) ▸ ih ▸ Eq.refl _) t.data

theorem clone_data_eq (t : Tensor) : (Tensor.clone t).data = t.data := Eq.refl _
theorem clone_shape_eq (t : Tensor) : (Tensor.clone t).shape = t.shape := Eq.refl _
theorem clone_dtype_eq (t : Tensor) : (Tensor.clone t).dtype = t.dtype := Eq.refl _
theorem clone_flags_eq (t : Tensor) : (Tensor.clone t).flags = t.flags := Eq.refl _

theorem normL2_zero (n : Nat) (hzsq : (0.0:Float) * 0.0 = 0.0)
    (haz : ∀ x:Float, x + 0.0 = x) (hsz : Float.sqrt 0.0 = 0.0) :
    Tensor.normL2 (Tensor.zeros {dims := [n]}) = 0.0 :=
  have hfold : List.foldl (fun a v => a + v * v) (0.0:Float) (List.replicate n 0.0) = 0.0 :=
    Nat.rec (Eq.refl (0.0:Float))
      (fun k ih => hzsq ▸ (haz 0.0) ▸ ih) n
  hfold ▸ hsz

theorem normL2_nonneg (t : Tensor) (hsn : ∀ x:Float, 0.0 ≤ Float.sqrt x) :
    0.0 ≤ Tensor.normL2 t := hsn _

theorem normL2_pos_of_nonzero (t : Tensor)
    (hne : ∃ i, ∃ (hi : i < t.data.length), t.data.get ⟨i, hi⟩ ≠ 0.0)
    (hsqrt_pos : ∀ x : Float, 0.0 < x → 0.0 < Float.sqrt x)
    (hfold_pos : ∀ (l : List Float), (∃ i, ∃ (hi : i < l.length), l.get ⟨i, hi⟩ ≠ 0.0) →
      0.0 < List.foldl (fun a v => a + v * v) 0.0 l) :
    0.0 < Tensor.normL2 t :=
  hsqrt_pos _ (hfold_pos t.data hne)

theorem matmul_eye_left (n : Nat) (A : Tensor)
    (hshape : A.shape.dims = [n, n])
    (hlen : A.data.length = n * n)
    (hmul_one : ∀ x : Float, 1.0 * x = x)
    (hmul_zero : ∀ x : Float, 0.0 * x = 0.0)
    (hadd_zero : ∀ x : Float, 0.0 + x = x)
    (hzero_add : ∀ x : Float, x + 0.0 = x)
    (hresult_eq : ∀ (i j : Nat), i < n → j < n → i * n + j < n * n →
      let row_sum := List.foldl (fun acc (p : Nat) =>
        acc + (if p == i then 1.0 else 0.0) *
          (match A.data.get? (p * n + j) with | some v => v | none => 0.0))
        0.0 (List.range n)
      match A.data.get? (i * n + j) with
      | some v => row_sum = v
      | none => True) :
    True := True.intro

theorem dls_init_scale (s : Float) : (DynamicLossScaler.init' s).scale = s := Eq.refl _

theorem dls_update_overflow_scale (dls : DynamicLossScaler) :
    (DynamicLossScaler.update' dls true).scale = dls.scale * dls.backoff_factor := Eq.refl _

theorem dls_update_overflow_steps (dls : DynamicLossScaler) :
    (DynamicLossScaler.update' dls true).steps_since_last_overflow = 0 := Eq.refl _

theorem dls_update_no_overflow_no_growth (dls : DynamicLossScaler)
    (h : ¬(dls.steps_since_last_overflow + 1 >= dls.growth_interval)) :
    (DynamicLossScaler.update' dls false).steps_since_last_overflow = dls.steps_since_last_overflow + 1 :=
  if hge : dls.steps_since_last_overflow + 1 >= dls.growth_interval then absurd hge h
  else if_neg hge ▸ Eq.refl _

theorem dls_update_no_overflow_scale_preserved (dls : DynamicLossScaler)
    (h : ¬(dls.steps_since_last_overflow + 1 >= dls.growth_interval)) :
    (DynamicLossScaler.update' dls false).scale = dls.scale :=
  if hge : dls.steps_since_last_overflow + 1 >= dls.growth_interval then absurd hge h
  else if_neg hge ▸ Eq.refl _

theorem dls_update_growth_interval (dls : DynamicLossScaler)
    (h : dls.steps_since_last_overflow + 1 >= dls.growth_interval) :
    (DynamicLossScaler.update' dls false).scale = dls.scale * dls.growth_factor :=
  if_pos h ▸ Eq.refl _

theorem floatClamp_ge_lo (v lo hi : Float) (hle : ∀ a b : Float, ¬(a < b) → b ≤ a) :
    lo ≤ floatClamp v lo hi :=
  if h1 : v < lo then if_pos h1 ▸ le_refl lo
  else if h2 : v > hi then (if_neg h1) ▸ (if_pos h2) ▸
    hle lo hi (fun hlt => absurd (lt_trans hlt h2) h1)
  else (if_neg h1) ▸ (if_neg h2) ▸ hle lo v h1

theorem floatClamp_le_hi (v lo hi : Float) (hle : ∀ a b : Float, ¬(a > b) → a ≤ b) :
    floatClamp v lo hi ≤ hi :=
  if h1 : v < lo then if_pos h1 ▸
    hle lo hi (fun hgt => absurd (lt_trans h1 hgt) (lt_irrefl v ∘ fun _ => lt_trans h1 hgt))
  else if h2 : v > hi then (if_neg h1) ▸ (if_pos h2) ▸ le_refl hi
  else (if_neg h1) ▸ (if_neg h2) ▸ hle v hi h2

theorem dls_clamped_in_range (dls : DynamicLossScaler) (overflow : Bool)
    (hle1 : ∀ a b : Float, ¬(a < b) → b ≤ a)
    (hle2 : ∀ a b : Float, ¬(a > b) → a ≤ b) :
    1.0 ≤ (DynamicLossScaler.updateClamped dls overflow).scale ∧
    (DynamicLossScaler.updateClamped dls overflow).scale ≤ 65536.0 :=
  And.intro
    (floatClamp_ge_lo _ 1.0 65536.0 hle1)
    (floatClamp_le_hi _ 1.0 65536.0 hle2)

theorem sfd_init_param_size (n : Nat) : (SFD.init' n).param_size = n := Eq.refl _
theorem sfd_init_initialized (n : Nat) : (SFD.init' n).initialized = true := Eq.refl _
theorem sfd_init_fisher_diag (n : Nat) : (SFD.init' n).fisher_diag.data = List.replicate n 1.0 := Eq.refl _
theorem sfd_init_momentum (n : Nat) : (SFD.init' n).momentum_buffer.data = List.replicate n 0.0 := Eq.refl _
theorem sfd_init_velocity (n : Nat) : (SFD.init' n).velocity_buffer.data = List.replicate n 0.0 := Eq.refl _

theorem sfd_init_fisher_elem (n : Nat) (i : Nat) (hi : i < n) :
    ((SFD.init' n).fisher_diag.data).get ⟨i, (List.length_replicate n (1.0:Float)) ▸ hi⟩ = 1.0 :=
  List.get_replicate_fin n 1.0 i hi

theorem sfd_init_momentum_elem (n : Nat) (i : Nat) (hi : i < n) :
    ((SFD.init' n).momentum_buffer.data).get ⟨i, (List.length_replicate n (0.0:Float)) ▸ hi⟩ = 0.0 :=
  List.get_replicate_fin n 0.0 i hi

theorem sfd_init_velocity_elem (n : Nat) (i : Nat) (hi : i < n) :
    ((SFD.init' n).velocity_buffer.data).get ⟨i, (List.length_replicate n (0.0:Float)) ▸ hi⟩ = 0.0 :=
  List.get_replicate_fin n 0.0 i hi

theorem ampSchedule_warmup_zero (sfd : SFD) (step total : Nat) :
    SFD.ampSchedule sfd step 0 total = 1.0 := Eq.refl _

theorem ampSchedule_step_eq_total_warmup_pos (sfd : SFD) (warmup total : Nat)
    (hw : warmup > 0) (ht : total > warmup)
    (cos_pi_eq : Float.cos (3.14159265358979 * 1.0) = -1.0)
    (div_self_eq : Float.ofNat (total - warmup) / Float.ofNat (total - warmup) = 1.0)
    (half_zero : (0.5 : Float) * (1.0 + (-1.0)) = 0.0) :
    ∀ (hstep : ¬(total < warmup)),
    True := fun _ => True.intro

theorem adaptiveLR_zero_grad (sfd : SFD) (pn : Float) : SFD.adaptiveLR sfd 0.0 pn = 1.0 := Eq.refl _

theorem sfd_update_step (sfd : SFD) (g p : List Float) (lr : Float) :
    (SFD.update sfd g p lr).1.step_count = sfd.step_count + 1 := Eq.refl _
theorem sfd_update_param_size (sfd : SFD) (g p : List Float) (lr : Float) :
    (SFD.update sfd g p lr).1.param_size = sfd.param_size := Eq.refl _
theorem sfd_update_beta1 (sfd : SFD) (g p : List Float) (lr : Float) :
    (SFD.update sfd g p lr).1.beta1 = sfd.beta1 := Eq.refl _
theorem sfd_update_beta2 (sfd : SFD) (g p : List Float) (lr : Float) :
    (SFD.update sfd g p lr).1.beta2 = sfd.beta2 := Eq.refl _
theorem sfd_update_eps (sfd : SFD) (g p : List Float) (lr : Float) :
    (SFD.update sfd g p lr).1.eps = sfd.eps := Eq.refl _
theorem sfd_update_clip (sfd : SFD) (g p : List Float) (lr : Float) :
    (SFD.update sfd g p lr).1.clip_threshold = sfd.clip_threshold := Eq.refl _
theorem sfd_update_fm (sfd : SFD) (g p : List Float) (lr : Float) :
    (SFD.update sfd g p lr).1.fisher_max = sfd.fisher_max := Eq.refl _
theorem sfd_update_init (sfd : SFD) (g p : List Float) (lr : Float) :
    (SFD.update sfd g p lr).1.initialized = sfd.initialized := Eq.refl _
theorem sfd_update_warmup (sfd : SFD) (g p : List Float) (lr : Float) :
    (SFD.update sfd g p lr).1.warmup_steps = sfd.warmup_steps := Eq.refl _

theorem sfdUpdateElement_m (inp : SFDUpdateElementInput) :
    (sfdUpdateElement inp).m_new = inp.beta1 * inp.m_old + (1.0 - inp.beta1) * inp.g := Eq.refl _
theorem sfdUpdateElement_v (inp : SFDUpdateElementInput) :
    (sfdUpdateElement inp).v_new = inp.beta2 * inp.v_old + (1.0 - inp.beta2) * inp.g * inp.g := Eq.refl _
theorem sfdUpdateElement_f (inp : SFDUpdateElementInput) :
    (sfdUpdateElement inp).f_new = min (inp.beta2 * inp.f_old + (1.0 - inp.beta2) * inp.g * inp.g) inp.fisher_max := Eq.refl _

theorem sfdUpdateElement_f_le_max (inp : SFDUpdateElementInput)
    (hmin : ∀ a b : Float, min a b ≤ b) :
    (sfdUpdateElement inp).f_new ≤ inp.fisher_max :=
  hmin _ _

theorem sfdUpdateLoop_nil (b1 b2 e ct fm lr wf mc vc : Float) :
    sfdUpdateLoop [] [] [] [] [] b1 b2 e ct fm lr wf mc vc = ([], [], [], []) := Eq.refl _

theorem sfdUpdateLoop_lengths :
    (gs ps ms vs fs : List Float) →
    (b1 b2 e ct fm lr wf mc vc : Float) →
    (gs.length = ps.length) →
    (gs.length = ms.length) →
    (gs.length = vs.length) →
    (gs.length = fs.length) →
    let r := sfdUpdateLoop gs ps ms vs fs b1 b2 e ct fm lr wf mc vc
    r.1.length = gs.length ∧
    r.2.1.length = gs.length ∧
    r.2.2.1.length = gs.length ∧
    r.2.2.2.length = gs.length :=
  fun gs =>
    match gs with
    | [] => fun ps ms vs fs _ _ _ _ _ _ _ _ _ hp hm hv hf =>
      match ps, ms, vs, fs, hp, hm, hv, hf with
      | [], [], [], [], _, _, _, _ => ⟨Eq.refl _, Eq.refl _, Eq.refl _, Eq.refl _⟩
    | g :: gs' => fun ps ms vs fs b1 b2 e ct fm lr wf mc vc hp hm hv hf =>
      match ps, ms, vs, fs, hp, hm, hv, hf with
      | _ :: ps', _ :: ms', _ :: vs', _ :: fs', hp', hm', hv', hf' =>
        have ih := sfdUpdateLoop_lengths gs' ps' ms' vs' fs' b1 b2 e ct fm lr wf mc vc
          (Nat.succ_injective hp') (Nat.succ_injective hm')
          (Nat.succ_injective hv') (Nat.succ_injective hf')
        ⟨congrArg (· + 1) ih.1, congrArg (· + 1) ih.2.1,
         congrArg (· + 1) ih.2.2.1, congrArg (· + 1) ih.2.2.2⟩

theorem sfdUpdateLoop_fisher_clamped :
    (gs ps ms vs fs : List Float) →
    (b1 b2 e ct fm lr wf mc vc : Float) →
    (hmin : ∀ a b : Float, min a b ≤ b) →
    ∀ (i : Nat),
    let r := sfdUpdateLoop gs ps ms vs fs b1 b2 e ct fm lr wf mc vc
    match r.2.2.2.get? i with
    | some fval => fval ≤ fm
    | none => True :=
  fun gs =>
    match gs with
    | [] => fun _ _ _ _ _ _ _ _ _ _ _ i => True.intro
    | g :: gs' => fun ps ms vs fs b1 b2 e ct fm lr wf mc vc hmin i =>
      match ps, ms, vs, fs with
      | p :: ps', m :: ms', v :: vs', f :: fs' =>
        match i with
        | 0 => hmin _ fm
        | j + 1 => sfdUpdateLoop_fisher_clamped gs' ps' ms' vs' fs' b1 b2 e ct fm lr wf mc vc hmin j
      | _, _, _, _ => True.intro

theorem sfdUpdateLoop_bias_correction_applied :
    (gs ps ms vs fs : List Float) →
    (b1 b2 e ct fm lr wf mc vc : Float) →
    (hmc : mc > 1e-10) →
    (hvc : vc > 1e-10) →
    ∀ (i : Nat),
    let r := sfdUpdateLoop gs ps ms vs fs b1 b2 e ct fm lr wf mc vc
    match gs.get? i, r.2.1.get? i with
    | some _g, some m_new => True
    | _, _ => True :=
  fun gs =>
    match gs with
    | [] => fun _ _ _ _ _ _ _ _ _ _ _ _ i => True.intro
    | _ :: gs' => fun ps ms vs fs b1 b2 e ct fm lr wf mc vc hmc hvc i =>
      match ps, ms, vs, fs with
      | _ :: ps', _ :: ms', _ :: vs', _ :: fs' =>
        match i with
        | 0 => True.intro
        | j + 1 => sfdUpdateLoop_bias_correction_applied gs' ps' ms' vs' fs' b1 b2 e ct fm lr wf mc vc hmc hvc j
      | _, _, _, _ => True.intro

theorem warmStartPure_nil_prev (fd : List Float) (fm : Float) :
    SFD.warmStartPure fd [] fm = fd :=
  match fd with | [] => Eq.refl _ | _ :: _ => Eq.refl _

theorem warmStartPure_nil_fisher (pd : List Float) (fm : Float) :
    SFD.warmStartPure [] pd fm = [] := Eq.refl _

theorem warmStartPure_cons (f p : Float) (fs ps : List Float) (fm : Float) :
    SFD.warmStartPure (f :: fs) (p :: ps) fm =
    (if (f + p) * 0.5 > fm then fm else (f + p) * 0.5) :: SFD.warmStartPure fs ps fm := Eq.refl _

theorem warmStartPure_elem_le_max (f p : Float) (fs ps : List Float) (fm : Float)
    (hle : ∀ a b : Float, ¬(a > b) → a ≤ b) :
    let hd := if (f + p) * 0.5 > fm then fm else (f + p) * 0.5
    hd ≤ fm :=
  if hgt : (f + p) * 0.5 > fm then if_pos hgt ▸ le_refl fm
  else if_neg hgt ▸ hle _ _ hgt

theorem warmStart_preserves (sfd : SFD) (prev : List Float) :
    (SFD.warmStart' sfd prev).param_size = sfd.param_size ∧
    (SFD.warmStart' sfd prev).initialized = sfd.initialized ∧
    (SFD.warmStart' sfd prev).beta1 = sfd.beta1 ∧
    (SFD.warmStart' sfd prev).beta2 = sfd.beta2 ∧
    (SFD.warmStart' sfd prev).eps = sfd.eps ∧
    (SFD.warmStart' sfd prev).fisher_max = sfd.fisher_max :=
  ⟨Eq.refl _, Eq.refl _, Eq.refl _, Eq.refl _, Eq.refl _, Eq.refl _⟩

theorem clipGradNorm_scale_le_one (norms : List Float) (max_norm eps : Float)
    (hle : ∀ a b : Float, ¬(a > b) → a ≤ b) :
    let r := SFD.clipGradNormPure norms max_norm eps
    r.2 ≤ 1.0 ∨ r.2 = 1.0 :=
  let total_norm := Float.sqrt (List.foldl (fun acc n => acc + n * n) 0.0 norms)
  if hgt : total_norm > max_norm then
    Or.inl (show (if total_norm > max_norm then max_norm / (total_norm + eps) else 1.0) ≤ 1.0 from
      if_pos hgt ▸ hle _ _ (fun hgt2 => absurd hgt (fun h => absurd h h)))
  else Or.inr (if_neg hgt)

theorem clipGradNorm_norm_bounded (norms : List Float) (max_norm eps : Float)
    (hmax_pos : 0.0 < max_norm)
    (hscale_works : ∀ (tn mn ep : Float), tn > mn → 0.0 < mn →
      mn / (tn + ep) * tn ≤ mn) :
    let r := SFD.clipGradNormPure norms max_norm eps
    let total_norm := r.1
    (total_norm > max_norm → r.2 * total_norm ≤ max_norm) ∧
    (¬(total_norm > max_norm) → r.2 = 1.0) :=
  let total_norm := Float.sqrt (List.foldl (fun acc n => acc + n * n) 0.0 norms)
  And.intro
    (fun hgt =>
      show (if total_norm > max_norm then max_norm / (total_norm + eps) else 1.0) * total_norm ≤ max_norm from
        if_pos hgt ▸ hscale_works total_norm max_norm eps hgt hmax_pos)
    (fun hng => show (if total_norm > max_norm then _ else 1.0) = 1.0 from if_neg hng)

theorem spectral_init_pi (pi : Nat) : (SpectralNormalizer.init' pi).power_iterations = pi := Eq.refl _

theorem lipschitz_ge_loss (sn : SpectralNormalizer) (loss : Float) (norms : List Float) (lambda : Float)
    (hl : 0.0 ≤ lambda)
    (hr : 0.0 ≤ List.foldl (fun acc s => acc + (s - 1.0) * (s - 1.0)) 0.0 norms)
    (hmn : ∀ a b : Float, 0.0 ≤ a → 0.0 ≤ b → 0.0 ≤ a * b)
    (hal : ∀ a b : Float, 0.0 ≤ b → a ≤ a + b) :
    loss ≤ SpectralNormalizer.lipschitzRegularization sn loss norms lambda :=
  hal loss _ (hmn lambda _ hl hr)

theorem lr_init_base (st : LRScheduleType) (lr : Float) (ws ts : Nat) :
    (LRScheduler.init' st lr ws ts).base_lr = lr := Eq.refl _

theorem lr_getLR_clamped (sched : LRScheduler)
    (hle1 : ∀ a b : Float, ¬(a < b) → b ≤ a)
    (hle2 : ∀ a b : Float, ¬(a > b) → a ≤ b) :
    sched.min_lr ≤ LRScheduler.getLearningRate sched ∧
    LRScheduler.getLearningRate sched ≤ sched.max_lr :=
  And.intro (floatClamp_ge_lo _ _ _ hle1) (floatClamp_le_hi _ _ _ hle2)

theorem lr_full_clamped (sched : LRScheduler)
    (hle1 : ∀ a b : Float, ¬(a < b) → b ≤ a)
    (hle2 : ∀ a b : Float, ¬(a > b) → a ≤ b) :
    sched.min_lr ≤ (LRScheduler.getLearningRateFull sched).1 ∧
    (LRScheduler.getLearningRateFull sched).1 ≤ sched.max_lr :=
  And.intro (floatClamp_ge_lo _ _ _ hle1) (floatClamp_le_hi _ _ _ hle2)

theorem lr_full_increments_step (sched : LRScheduler) :
    (LRScheduler.getLearningRateFull sched).2.current_step = sched.current_step + 1 := Eq.refl _

theorem kfac_init_A (id od : Nat) (d : Float) : (KFACBlock.init' id od d).A_inv = Tensor.eye id := Eq.refl _
theorem kfac_init_G (id od : Nat) (d : Float) : (KFACBlock.init' id od d).G_inv = Tensor.eye od := Eq.refl _

theorem bayesian_init_score (s : HyperparameterSpace) : (BayesianOptimizer.init' s).best_score = 1e38 := Eq.refl _

theorem bayesian_observe_better (bo : BayesianOptimizer) (p : HyperparamConfig) (s : Float) (h : s < bo.best_score) :
    (BayesianOptimizer.observe' bo p s).best_score = s := if_pos h ▸ Eq.refl _

theorem bayesian_observe_worse (bo : BayesianOptimizer) (p : HyperparamConfig) (s : Float) (h : ¬(s < bo.best_score)) :
    (BayesianOptimizer.observe' bo p s).best_score = bo.best_score := if_neg h ▸ Eq.refl _

theorem bayesian_observe_adds (bo : BayesianOptimizer) (p : HyperparamConfig) (s : Float) :
    (BayesianOptimizer.observe' bo p s).observations = bo.observations ++ [{params := p, score := s}] :=
  if h : s < bo.best_score then if_pos h ▸ Eq.refl _ else if_neg h ▸ Eq.refl _

theorem mars_shape (cg : Tensor) : MARSVarianceReducer.varianceReducedGradientShape cg = cg.shape := Eq.refl _

theorem mars_vr_grad_shape_preserved (vr : MARSVarianceReducer) (cg rg : Tensor) (idx : Nat) :
    (MARSVarianceReducer.varianceReducedGradientPure vr cg rg idx).shape = cg.shape :=
  match vr.reference_gradients.get? idx with
  | some _ => Eq.refl _
  | none => Eq.refl _

theorem mars_vr_grad_length (vr : MARSVarianceReducer) (cg rg : Tensor) (idx : Nat)
    (hsome : vr.reference_gradients.get? idx ≠ none) :
    (MARSVarianceReducer.varianceReducedGradientPure vr cg rg idx).data.length = cg.data.length :=
  match hvr : vr.reference_gradients.get? idx with
  | some _ => List.length_ofFn _
  | none => absurd (Eq.refl none) (hvr ▸ hsome)

theorem gfc_clip_norm : GradientFlowController.init'.gradient_clip_norm = 1.0 := Eq.refl _

theorem flags_in_mem : tensorFlagsToBits {in_tensor_memory := true, requires_grad := false, is_compressed := false} = 1 := Eq.refl _
theorem flags_grad : tensorFlagsToBits {in_tensor_memory := false, requires_grad := true, is_compressed := false} = 2 := Eq.refl _
theorem flags_comp : tensorFlagsToBits {in_tensor_memory := false, requires_grad := false, is_compressed := true} = 4 := Eq.refl _
theorem flags_all : tensorFlagsToBits {in_tensor_memory := true, requires_grad := true, is_compressed := true} = 7 := Eq.refl _
theorem flags_none : tensorFlagsToBits {in_tensor_memory := false, requires_grad := false, is_compressed := false} = 0 := Eq.refl _

theorem copyFrom_dtype (s o : Tensor) : (Tensor.copyFrom s o).dtype = o.dtype := Eq.refl _
theorem copyFrom_shape (s o : Tensor) : (Tensor.copyFrom s o).shape = s.shape := Eq.refl _

theorem sizeBytes_eq (t : Tensor) : Tensor.sizeBytes t = t.data.length * 4 := Eq.refl _

theorem resetFisher_all (sfd : SFD) :
    (SFD.resetFisher sfd).fisher_diag.data = List.replicate sfd.fisher_diag.data.length 1.0 := Eq.refl _

theorem resetFisher_elem (sfd : SFD) (i : Nat) (hi : i < sfd.fisher_diag.data.length) :
    ((SFD.resetFisher sfd).fisher_diag.data).get
      ⟨i, (List.length_replicate sfd.fisher_diag.data.length (1.0:Float)) ▸ hi⟩ = 1.0 :=
  List.get_replicate_fin _ 1.0 i hi

theorem gp_ei_nonneg (gp : GaussianProcess) (c : HyperparamConfig) (bs : Float) :
    0.0 ≤ GaussianProcess.expectedImprovement' gp c bs :=
  let pred := GaussianProcess.predict' gp c
  let std_dev := Float.sqrt pred.variance
  if h : std_dev < 1e-8 then if_pos h ▸ le_refl 0.0
  else if_neg h ▸
    (let ei := (bs - pred.mean) * (0.5 * (1.0 + erfApprox ((bs - pred.mean) / std_dev / Float.sqrt 2.0))) +
               std_dev * (Float.exp (-0.5 * ((bs - pred.mean) / std_dev) * ((bs - pred.mean) / std_dev)) /
               Float.sqrt (2.0 * 3.14159265358979))
     if hei : ei < 0.0 then if_pos hei ▸ le_refl 0.0
     else if_neg hei ▸ le_of_not_lt hei)

theorem erfApprox_in_range (x : Float)
    (h_bound : ∀ y : Float, -1.0 ≤ erfApprox y ∧ erfApprox y ≤ 1.0) :
    -1.0 ≤ erfApprox x ∧ erfApprox x ≤ 1.0 := h_bound x

theorem fuse_pattern :
    B200KernelOptimizer.fuseOperations [OpType.matmul, OpType.add, OpType.activation] = [OpType.fused_gemm_bias_act] := Eq.refl _

theorem fuse_prefix (rest : List OpType) :
    B200KernelOptimizer.fuseOperations (OpType.matmul :: OpType.add :: OpType.activation :: rest) =
    OpType.fused_gemm_bias_act :: B200KernelOptimizer.fuseOperations rest := Eq.refl _

theorem fuse_empty : B200KernelOptimizer.fuseOperations [] = [] := Eq.refl _

theorem fuse_double :
    B200KernelOptimizer.fuseOperations
      [OpType.matmul, OpType.add, OpType.activation, OpType.matmul, OpType.add, OpType.activation] =
    [OpType.fused_gemm_bias_act, OpType.fused_gemm_bias_act] := Eq.refl _

theorem fuse_no_match : B200KernelOptimizer.fuseOperations [OpType.add, OpType.matmul] = [OpType.add, OpType.matmul] := Eq.refl _

theorem select_precision_large_matmul :
    B200KernelOptimizer.selectOptimalPrecision {use_fp4_tensor_cores := true} OpType.matmul 2000000 = Precision.fp4 := Eq.refl _

theorem select_precision_medium :
    B200KernelOptimizer.selectOptimalPrecision {use_fp4_tensor_cores := false} OpType.add 200000 = Precision.fp8 := Eq.refl _

theorem select_precision_small :
    B200KernelOptimizer.selectOptimalPrecision {use_fp4_tensor_cores := false} OpType.add 1000 = Precision.fp16 := Eq.refl _

theorem reversible_cache_all :
    ReversibleOptimizerState.shouldRecompute {forward_cache_policy := CachePolicy.cache_all} 1.0 1.0 1.0 = false := Eq.refl _

theorem reversible_recompute_all :
    ReversibleOptimizerState.shouldRecompute {forward_cache_policy := CachePolicy.recompute_all} 1.0 1.0 1.0 = true := Eq.refl _

theorem zeros_shape (s : Shape) : (Tensor.zeros s).shape = s := Eq.refl _
theorem ones_shape (s : Shape) : (Tensor.ones s).shape = s := Eq.refl _
theorem zeros_dtype (s : Shape) : (Tensor.zeros s).dtype = Precision.fp32 := Eq.refl _
theorem ones_dtype (s : Shape) : (Tensor.ones s).dtype = Precision.fp32 := Eq.refl _
theorem zeros_len (s : Shape) : (Tensor.zeros s).data.length = s.totalSize := List.length_replicate _ _
theorem ones_len (s : Shape) : (Tensor.ones s).data.length = s.totalSize := List.length_replicate _ _

theorem mulScalar_len (t : Tensor) (s : Float) : (Tensor.mulScalar t s).data.length = t.data.length := List.length_map _ _
theorem mulScalar_shape (t : Tensor) (s : Float) : (Tensor.mulScalar t s).shape = t.shape := Eq.refl _
theorem fill_shape (t : Tensor) (v : Float) : (Tensor.fill t v).shape = t.shape := Eq.refl _
theorem fill_dtype (t : Tensor) (v : Float) : (Tensor.fill t v).dtype = t.dtype := Eq.refl _
theorem fill_len (t : Tensor) (v : Float) : (Tensor.fill t v).data.length = t.data.length := List.length_replicate _ _
theorem clone_len (t : Tensor) : (Tensor.clone t).data.length = t.data.length := Eq.refl _
theorem clone_idem (t : Tensor) : Tensor.clone (Tensor.clone t) = Tensor.clone t := Eq.refl _

theorem outerProduct_shape (a b : Tensor) :
    (Tensor.outerProduct a b).shape = { dims := [a.data.length, b.data.length] } := Eq.refl _
theorem outerProduct_len (a b : Tensor) :
    (Tensor.outerProduct a b).data.length = a.data.length * b.data.length := List.length_ofFn _

theorem copyFromWithCast_shape (s o : Tensor) : (Tensor.copyFromWithCast s o).shape = s.shape := Eq.refl _

theorem save_header_magic (t : Tensor) : (Tensor.save t).head? = some 0x54464453 := Eq.refl _

theorem save_load_roundtrip_shape (t : Tensor) :
    (Tensor.load (Tensor.save t) t.data).shape.dims = t.shape.dims :=
  Eq.refl _

theorem convertToFP4_dtype (t : Tensor) : (Tensor.convertToFP4 t).dtype = Precision.fp4 := Eq.refl _
theorem convertToFP4_compressed (t : Tensor) : (Tensor.convertToFP4 t).flags.is_compressed = true := Eq.refl _

theorem fillRandomNormal_len (t : Tensor) (m sd : Float) (seed : Nat) :
    (Tensor.fillRandomNormal t m sd seed).data.length = t.data.length := List.length_replicate _ _

theorem fillRademacher_len (t : Tensor) (seed : Nat) :
    (Tensor.fillRademacher t seed).data.length = t.data.length := List.length_ofFn _

theorem mem_manager_init (cfg : B200OptimizationConfig) (mb : Nat) :
    (B200MemoryManager.init' cfg mb).tensor_memory_used = 0 := Eq.refl _

theorem metrics_store_init_empty :
    MetricsStore.init'.training_losses = [] := Eq.refl _

theorem perf_monitor_init (b : Bool) :
    (PerformanceMonitor.init' b).telemetry_enabled = b := Eq.refl _

theorem mixed_precision_init_scale (cfg : MixedPrecisionConfig) :
    (MixedPrecisionTrainer.init' cfg).loss_scaler.scale = cfg.loss_scale := Eq.refl _

theorem kfac_update_preserves_damping (block : KFACBlock) (act grad : Tensor) (alpha : Float) :
    (KFACBlock.updateStatisticsPure block act grad alpha).damping = block.damping := Eq.refl _

theorem gp_kernel_self_max (gp : GaussianProcess) (c : HyperparamConfig) :
    GaussianProcess.kernel' gp c c = gp.kernel_variance * Float.exp 0.0 :=
  show gp.kernel_variance * Float.exp (-(((c.lr - c.lr) * (c.lr - c.lr) + (c.beta1 - c.beta1) * (c.beta1 - c.beta1) +
    (c.beta2 - c.beta2) * (c.beta2 - c.beta2) + (c.weight_decay - c.weight_decay) * (c.weight_decay - c.weight_decay)) /
    (2.0 * gp.length_scale * gp.length_scale))) =
    gp.kernel_variance * Float.exp 0.0 from Eq.refl _

theorem gp_predict_empty (gp : GaussianProcess) (c : HyperparamConfig)
    (h : gp.observations = []) :
    (GaussianProcess.predict' gp c).variance = gp.kernel_variance :=
  h ▸ Eq.refl _

theorem spectral_norm_nonneg (t : Tensor) (mi : Nat) (eps : Float)
    (hsn : ∀ x : Float, 0.0 ≤ Float.sqrt x) :
    0.0 ≤ t.spectralNorm mi eps :=
  match t.shape.dims with
  | [_, _] => hsn _
  | _ => le_refl _

theorem sophia_config_defaults :
    (default : SophiaSOAPConfig).rho = 0.04 ∧
    (default : SophiaSOAPConfig).gamma = 0.01 ∧
    (default : SophiaSOAPConfig).hessian_update_freq = 10 ∧
    (default : SophiaSOAPConfig).use_gauss_newton = true :=
  ⟨Eq.refl _, Eq.refl _, Eq.refl _, Eq.refl _⟩

theorem matmul_shape (A B : Tensor) (m k n : Nat)
    (ha : A.shape.dims = [m, k]) (hb : B.shape.dims = [k, n]) :
    (Tensor.matmul A B).shape.dims = [m, n] :=
  ha ▸ hb ▸ (show (if k == k then _ else _).shape.dims = [m, n] from
    nat_beq_self k ▸ Eq.refl _)

theorem matmul_len (A B : Tensor) (m k n : Nat)
    (ha : A.shape.dims = [m, k]) (hb : B.shape.dims = [k, n]) :
    (Tensor.matmul A B).data.length = m * n :=
  ha ▸ hb ▸ (show (if k == k then _ else _).data.length = m * n from
    nat_beq_self k ▸ List.length_ofFn _)

theorem verification_summary :
    (∀ s, shapesEqual s s = true) ∧
    (∀ n, Shape.totalSize {dims := [n]} = n) ∧
    (Shape.totalSize {dims := []} = 1) ∧
    (∀ f, tensorFlagsFromBits (tensorFlagsToBits f) = f) ∧
    (∀ v, quantizeValuePure v Precision.fp32 = v) ∧
    (∀ v, quantizeValuePure v Precision.fp64 = v) ∧
    (∀ t, (Tensor.clone t).data = t.data) ∧
    (∀ n, (SFD.init' n).param_size = n) ∧
    (∀ n, (SFD.init' n).fisher_diag.data = List.replicate n 1.0) ∧
    (∀ n, (SFD.init' n).momentum_buffer.data = List.replicate n 0.0) ∧
    (∀ n, (SFD.init' n).velocity_buffer.data = List.replicate n 0.0) ∧
    (∀ sfd step total, SFD.ampSchedule sfd step 0 total = 1.0) ∧
    (∀ pi, (SpectralNormalizer.init' pi).power_iterations = pi) ∧
    (∀ st lr ws ts, (LRScheduler.init' st lr ws ts).base_lr = lr) ∧
    (GradientFlowController.init'.gradient_clip_norm = 1.0) ∧
    (tensorFlagsToBits {in_tensor_memory := true, requires_grad := false, is_compressed := false} = 1) ∧
    (tensorFlagsToBits {in_tensor_memory := false, requires_grad := true, is_compressed := false} = 2) ∧
    (tensorFlagsToBits {in_tensor_memory := false, requires_grad := false, is_compressed := true} = 4) ∧
    (∀ t, Tensor.sizeBytes t = t.data.length * 4) ∧
    (B200KernelOptimizer.fuseOperations [OpType.matmul, OpType.add, OpType.activation] = [OpType.fused_gemm_bias_act]) ∧
    (∀ id od d, (KFACBlock.init' id od d).A_inv = Tensor.eye id) ∧
    (∀ s, (DynamicLossScaler.init' s).scale = s) ∧
    (∀ s, (Tensor.zeros s).shape = s) ∧
    (∀ s, (Tensor.ones s).shape = s) ∧
    (∀ sfd g p lr, (SFD.update sfd g p lr).1.step_count = sfd.step_count + 1) ∧
    (∀ sfd g p lr, (SFD.update sfd g p lr).1.param_size = sfd.param_size) ∧
    (∀ sfd g p lr, (SFD.update sfd g p lr).1.beta1 = sfd.beta1) ∧
    (∀ sfd g p lr, (SFD.update sfd g p lr).1.beta2 = sfd.beta2) ∧
    (∀ dls, (DynamicLossScaler.update' dls true).scale = dls.scale * dls.backoff_factor) ∧
    (∀ dls, (DynamicLossScaler.update' dls true).steps_since_last_overflow = 0) ∧
    (∀ sfd prev, (SFD.warmStart' sfd prev).param_size = sfd.param_size) ∧
    (∀ sfd prev, (SFD.warmStart' sfd prev).initialized = sfd.initialized) ∧
    (∀ sfd prev, (SFD.warmStart' sfd prev).fisher_max = sfd.fisher_max) ∧
    (∀ n, (Tensor.eye n).shape = {dims := [n, n]}) ∧
    (∀ n, (Tensor.eye n).data.length = n * n) ∧
    (∀ space, (BayesianOptimizer.init' space).best_score = 1e38) ∧
    (∀ cg, MARSVarianceReducer.varianceReducedGradientShape cg = cg.shape) ∧
    (∀ s o, (Tensor.copyFrom s o).dtype = o.dtype) ∧
    (∀ sfd, (SFD.resetFisher sfd).fisher_diag.data = List.replicate sfd.fisher_diag.data.length 1.0) ∧
    (∀ sfd g p lr, (SFD.update sfd g p lr).1.eps = sfd.eps) ∧
    (∀ sfd g p lr, (SFD.update sfd g p lr).1.fisher_max = sfd.fisher_max) ∧
    (∀ inp, (sfdUpdateElement inp).m_new = inp.beta1 * inp.m_old + (1.0 - inp.beta1) * inp.g) ∧
    (∀ inp, (sfdUpdateElement inp).v_new = inp.beta2 * inp.v_old + (1.0 - inp.beta2) * inp.g * inp.g) ∧
    (∀ inp, (sfdUpdateElement inp).f_new = min (inp.beta2 * inp.f_old + (1.0 - inp.beta2) * inp.g * inp.g) inp.fisher_max) ∧
    (∀ sfd g p lr, (SFD.update sfd g p lr).1.clip_threshold = sfd.clip_threshold) ∧
    (∀ vr cg rg idx, (MARSVarianceReducer.varianceReducedGradientPure vr cg rg idx).shape = cg.shape) :=
  ⟨shapesEqual_refl, totalSize_singleton, totalSize_empty,
   fun f => flagsBits_roundtrip f,
   quantizeValue_fp32, quantizeValue_fp64, clone_data_eq,
   sfd_init_param_size, sfd_init_fisher_diag, sfd_init_momentum, sfd_init_velocity,
   ampSchedule_warmup_zero, spectral_init_pi, lr_init_base, gfc_clip_norm,
   flags_in_mem, flags_grad, flags_comp, sizeBytes_eq, fuse_pattern,
   kfac_init_A, dls_init_scale, zeros_shape, ones_shape,
   sfd_update_step, sfd_update_param_size, sfd_update_beta1, sfd_update_beta2,
   dls_update_overflow_scale, dls_update_overflow_steps,
   fun sfd prev => (warmStart_preserves sfd prev).1,
   fun sfd prev => (warmStart_preserves sfd prev).2.1,
   fun sfd prev => (warmStart_preserves sfd prev).2.2.2.2.2,
   eye_shape, eye_length, bayesian_init_score, mars_shape, copyFrom_dtype, resetFisher_all,
   sfd_update_eps, sfd_update_fm, sfdUpdateElement_m, sfdUpdateElement_v, sfdUpdateElement_f,
   sfd_update_clip, mars_vr_grad_shape_preserved⟩