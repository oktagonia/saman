library(saman)

coord_fun = function(t) {
  c(2 * cos(t[1]), sin(t[1]))
}

pushforward_fun = function(t) {
  matrix(c(-2 * sin(t[1]), cos(t[1])), nrow = 2, ncol = 1)
}

ellipse = saman.manifold(
  n = 2,
  k = 1,
  vf_bound = 2.5,
  lower = c(0.0),
  upper = c(2 * pi),
  coord = coord_fun,
  pushforward = pushforward_fun
)

samples = ellipse$sample(35)

plot(samples[, 1], samples[, 2], 
     asp = 1, 
     col = "red", 
     pch = 19, 
     main = "Uniform Sampling on Ellipse", 
     xlab = "x", 
     ylab = "y")

t_vals = seq(0, 2 * pi, length.out = 100)
lines(2 * cos(t_vals), sin(t_vals), col = "blue")