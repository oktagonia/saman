saman.manifold = function(n, k, vf_bound, lower, upper, coord, pushforward) {
  ptr = manifold_cpp(
    n = as.integer(n),
    k = as.integer(k),
    vf_bound = as.double(vf_bound),
    lower = as.numeric(lower),
    upper = as.numeric(upper),
    coord_fun = coord,
    pushforward_fun = pushforward
  )

  structure(
    list(
      ptr = ptr,
      n = manifold_n_cpp(ptr),
      k = manifold_k_cpp(ptr),
      vf_bound = manifold_vf_bound_cpp(ptr),
      coord = function(u) manifold_coord_cpp(ptr, as.numeric(u)),
      pushforward = function(u) manifold_pushforward_cpp(ptr, as.numeric(u)),
      metric = function(u) manifold_metric_cpp(ptr, as.numeric(u)),
      volume_form = function(u) manifold_volume_form_cpp(ptr, as.numeric(u)),
      sample = function(n_samples) manifold_sample_cpp(ptr, as.integer(n_samples))
    ),
    class = "manifold"
  )
}