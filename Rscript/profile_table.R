library(ggplot2)
library(dplyr)

internal_viz_rainforest <- function(plotdata, madata,
                                    type = "standard",
                                    study_labels = NULL, summary_label = NULL,
                                    study_table = NULL, summary_table = NULL, annotate_CI = FALSE,
                                    confidence_level = 0.95, col = "Blues", summary_col = "Blues",
                                    detail_level = 1,
                                    text_size = 3, xlab = "Effect", x_limit = NULL,
                                    x_trans_function = NULL, x_breaks = NULL) {
  n <- nrow(plotdata)
  k <- length(levels(plotdata$group))

  # weight of each study used to scale the height of each raindrop
  if(type %in% c("standard", "study_only")) {
    weight <- 1/(plotdata$se^2 + madata$summary_tau2[as.numeric(plotdata$group)])
  } else {
    weight <- 1/plotdata$se^2
  }
  plotdata$rel_weight <- weight/sum(weight)

  tick_size <- max(plotdata$rel_weight/(6 * max(plotdata$rel_weight)))
  tickdata <- data.frame(x = c(plotdata$x, plotdata$x), ID = c(plotdata$ID, plotdata$ID),
                         y = c(plotdata$ID + tick_size,
                               plotdata$ID - tick_size))

  # function ll constructs a likelihood raindrop of a study. Each raindrop is built out of
  # several distinct segments (to color shade the raindrop)
  ll <- function(x, max.range, max.weight) {
    # width of the region over which the raindop is built
    se.factor <- ceiling(stats::qnorm(1 - (1 - confidence_level)/2))
    width <- abs((x[1] - se.factor * x[2]) - (x[1] + se.factor * x[2]))

    # max.range is internally determined as the width of the broadest raindop.
    # The number of points to construct the raindrop is chosen proportional to
    # the ratio of the width of the raindrop and the max.range,
    # because slim/dense raindrops do not need as many support points as very broad ones.
    # Minimum is 200 points (for the case that width/max.range gets very small)
    length.out <- max(c(floor(1000 * width / max.range), 200))

    # Create sequence of points to construct the raindrop. The number of points is chosen by length.out
    # and can be changed by the user with the parameter detail_level.
    # At least 50 support points (for detail level << 1) per raindrop are chosen
    support <- seq(x[1] - se.factor * x[2], x[1] + se.factor * x[2], length.out = max(c(length.out * detail_level, 50)))

    # The values for the likelihood drop are determined: The likelihood for different hypothetical true values
    # minus likelihood for the observed value (i.e. the maximum likelihood) plus the confidence.level quantile of the chi square
    # distribution with one degree of freedeom divided by two.
    # Explanation: -2*(log(L(observed)/L(hypothetical))) is an LRT test and approx. chi^2 with 1 df and significance threshold
    # qchisq(confidence.level, df = 1).
    # That means by adding the confidence.level quantile of the chi square
    # distribution with one degree of freedom (the significance threshold) divided by two,
    # values with l_mu < 0 differ significantly from the observed value.
    threshold <- stats::qchisq(confidence_level, df = 1)/2
    l_mu <- log(stats::dnorm(x[1], mean = support, sd = x[2])) - log(stats::dnorm(x[1], mean = x[1], sd = x[2])) + threshold

    #scale raindrop such that it is proportional to the meta-analytic weight and has height smaller than 0.5
    l_mu <- l_mu/max(l_mu) * x[3]/max.weight * 0.45

    # Force raindrops of studies to have minimum height of 0.05 (i.e. approx. one tenth of the raindrop with maximum height)
    if(max(l_mu) < 0.05) {
      l_mu <- l_mu/max(l_mu) * 0.05
    }

    # mirror values for raindrop
    l_mu_mirror <- -l_mu

    # select only likelihood values that are equal or larger than zero,
    # i.e. values that also lie in the confidence interval (using normality assumption)
    sel <- which(l_mu >= 0)

    # Construct data.frame
    d <- data.frame("support" = c(support[sel], rev(support[sel])), "log_density" = c(l_mu[sel], rev(l_mu_mirror[sel])))

    # The number of segments for shading is chosen as follows: 40 segements times the detail_level per drop
    # as default. The minimum count of segments is 20 (If detail_level is << 1), with the exception that
    # if there are too few points for 20 segments then nrow(d)/4 is used (i.e. at least 4 points per segment)
    data.frame(d, "segment" = cut(d$support, max(c(40*detail_level), min(c(20, nrow(d)/4)))))
  }

  # compute the max range of all likelihood drops for function ll.
  max.range <- max(abs((plotdata$x + stats::qnorm(1 - (1 - confidence_level)/2) * plotdata$se) -
                         (plotdata$x - (stats::qnorm(1 - (1 - confidence_level)/2) * plotdata$se))))

  # computes all likelihood values and segments. The output is a list, where every element
  # constitutes one study raindop
  res <- apply(cbind(plotdata$x, plotdata$se, plotdata$rel_weight), 1,  FUN = function(x) {ll(x, max.range = max.range, max.weight = max(plotdata$rel_weight))})

  # name every list entry, i.e. raindrop, and add id column
  names(res) <- plotdata$ID
  for(i in 1:length(res)) {
    res[[i]] <- data.frame(res[[i]], .id = plotdata$ID[i])
  }

  # The prep.data function prepares the list of raindrops in three ways for plotting (shading of segments):
  # 1) the values are sorted by segments, such that the same segments of each raindrop are joined together
  # 2) segments are renamed with integer values from 1 to the number of segments per raindrop
  # 3) to draw smooth raindrops the values at the right hand boundary of each segment have to be the first
  # values at the left hand boundary of the next segment on the right.
  prep.data <- function(res) {
    res <- lapply(res, FUN = function(x) {x <- x[order(x$segment), ]})
    res <- lapply(res, FUN = function(x) {x$segment <- factor(x$segment, labels = 1:length(unique(x$segment))); x})
    res <- lapply(res, FUN = function(x) {
      seg_n <- length(unique(x$segment))
      first <- sapply(2:seg_n, FUN = function(n) {min(which(as.numeric(x$segment)==n))})
      last <-  sapply(2:seg_n, FUN = function(n) {max(which(as.numeric(x$segment)==n))})
      neighbor.top <-   x[c(stats::aggregate(support~segment, FUN = which.max, data=x)$support[1],
                            cumsum(stats::aggregate(support~segment, FUN = length, data=x)$support)[-c(seg_n-1, seg_n)] +
                              stats::aggregate(support~segment, FUN = which.max, data=x)$support[-c(1, seg_n)]), c("support", "log_density")]
      neighbor.bottom <-   x[c(stats::aggregate(support~segment, FUN = which.max, data=x)$support[1],
                               cumsum(stats::aggregate(support~segment, FUN = length, data=x)$support[-c(seg_n-1, seg_n)])+
                                 stats::aggregate(support~segment, FUN = which.max, data=x)$support[-c(1, seg_n)]) + 1, c("support", "log_density")]
      x[first, c("support", "log_density")] <- neighbor.top
      x[last, c("support", "log_density")] <- neighbor.bottom
      x
    }
    )
    res
  }
  res <- prep.data(res)

  # merge the list of raindops in one dataframe for plotting
  res <- do.call(rbind, res)

  # set limits and breaks for the y axis and construct summary diamond (for type standard and sensitivity)
  if(type %in% c("standard", "sensitivity", "cumulative")) {
    y_limit <- c(min(plotdata$ID) - 3, max(plotdata$ID) + 1.5)
    y_tick_names <- c(as.vector(study_labels), as.vector(summary_label))[order(c(plotdata$ID, madata$ID), decreasing = T)]
    y_breaks <- sort(c(plotdata$ID, madata$ID), decreasing = T)
    summarydata <- data.frame("x.diamond" = c(madata$summary_es - stats::qnorm(1 - (1 - confidence_level) / 2, 0, 1) * madata$summary_se,
                                              madata$summary_es,
                                              madata$summary_es + stats::qnorm(1 - (1 - confidence_level) / 2, 0, 1) * madata$summary_se,
                                              madata$summary_es),
                              "y.diamond" = c(madata$ID,
                                              madata$ID + 0.3,
                                              madata$ID,
                                              madata$ID - 0.3),
                              "diamond_group" = rep(1:k, times = 4)
    )
  } else {
    y_limit <- c(min(plotdata$ID) - 1, max(plotdata$ID) + 1.5)
    y_tick_names <- plotdata$labels[order(plotdata$ID, decreasing = T)]
    y_breaks <- sort(plotdata$ID, decreasing = T)
  }

  # set limits for the x axis if none are supplied
  if(is.null(x_limit)) {
    x_limit <- c(range(c(plotdata$x_min, plotdata$x_max))[1] - diff(range(c(plotdata$x_min, plotdata$x_max)))*0.05,
                 range(c(plotdata$x_min, plotdata$x_max))[2] + diff(range(c(plotdata$x_min, plotdata$x_max)))*0.05)
  }

  # To shade all segments of each raindop symmetrically the min abs(log_density) per raindrop is used
  # as aesthetic to fill the segments. This is necessary because otherwise the first log_density value per
  # segment would be used leading to asymmetrical shading
  min.ld <- stats::aggregate(log_density ~ segment + .id, FUN  = function(x) {min(abs(x))}, data = res)
  names(min.ld) <- c("segment", ".id", "min_log_density")
  res <- merge(res, min.ld, sort = F)

  # Set Color palette for shading
  if(type != "summary_only") {
    if(!(col %in% c("Blues", "Greys", "Oranges", "Greens", "Reds", "Purples"))) {
      warning("Supported arguments for col for rainforest plots are Blues, Greys, Oranges, Greens, Reds, and Purples. Blues is used.")
      col <- "Blues"
    }
    col <- RColorBrewer::brewer.pal(n = 9, name = col)
    if(summary_col %in% c("Blues", "Greys", "Oranges", "Greens", "Reds", "Purples")) {
      summary_col <- RColorBrewer::brewer.pal(n = 9, name = summary_col)[9]
    }
  } else {
    if(type == "summary_only") {
      if(!(summary_col %in% c("Blues", "Greys", "Oranges", "Greens", "Reds", "Purples"))) {
        warning("Supported arguments for summary_col for summary-only rainforest plots are Blues, Greys, Oranges, Greens, Reds, and Purples. Blues is used.")
        summary_col <- "Blues"
      }
      summary_col <- RColorBrewer::brewer.pal(n = 9, name = col)
      col <- summary_col
    }
  }

  # Set plot margins. If table is aligned on the left, no y axis breaks and ticks are plotted
  l <- 5.5
  r <- 11
  if(annotate_CI == TRUE) {
    r <- 1
  }
  if(!is.null(study_table) || !is.null(summary_table)) {
    l <- 1
    y_tick_names <- NULL
    y_breaks <- NULL
  }
  # workaround for "Undefined global functions or variables" Note in R CMD check while using ggplot2.
  support <- NULL
  segment <- NULL
  min_log_density <- NULL
  log_density <- NULL
  .id <-
    x.diamond <- NULL
  y.diamond <- NULL
  diamond_group <- NULL
  x <- NULL
  y <- NULL
  x_min <- NULL
  x_max <- NULL
  ID <- NULL

  # Create Rainforest plot
  p <-
    ggplot(data = res, aes(y = .id, x = support)) +
    geom_errorbarh(data = plotdata, col = col[1], aes(xmin = x_min, xmax = x_max, y = ID, height = 0), inherit.aes = FALSE) +
    geom_polygon(data = res, aes(x = support, y = as.numeric(.id) + log_density,
                                 color = min_log_density, fill = min_log_density,
                                 group = paste(.id, segment)), size = 0.1) +
    geom_line(data = tickdata, aes(x = x, y = y, group = ID), col = "grey", size = 1)
  # geom_errorbarh(data = plotdata, col = "grey", aes(x = x, xmin = x_min, xmax = x_max, y = ID, height = 0))
  if(type %in% c("standard", "sensitivity", "cumulative")) {
    p <- p + geom_polygon(data = summarydata, aes(x = x.diamond, y = y.diamond, group = diamond_group), color="black", fill = summary_col, size = 0.1)
  }
  p <- p +
    scale_fill_gradient(high = col[9], low = col[3], guide = FALSE) +
    scale_color_gradient(high = col[9], low = col[3], guide = FALSE) +
    geom_vline(xintercept = 0, linetype = 2) +
    scale_y_continuous(name = "",
                       breaks = y_breaks,
                       labels = y_tick_names) +
    coord_cartesian(xlim = x_limit, ylim = y_limit, expand = F)
  if(!is.null(x_trans_function)) {
    if(is.null(x_breaks)) {
      p <- p +
        scale_x_continuous(name = xlab,
                           labels = function(x) {round(x_trans_function(x), 3)})
    } else {
      p <- p +
        scale_x_continuous(name = xlab,
                           labels = function(x) {round(x_trans_function(x), 3)},
                           breaks = x_breaks)
    }
  } else {
    if(is.null(x_breaks)) {
      p <- p +
        scale_x_continuous(name = xlab)
    } else {
      p <- p +
        scale_x_continuous(breaks = x_breaks,
                           name = xlab)
    }
  }
  p <- p +
    theme_bw() +
    theme(text = element_text(size = 1/0.352777778*text_size),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.x = element_line("grey"),
          panel.grid.minor.x = element_line("grey"),
          plot.margin = margin(t = 5.5, r = r, b = 5.5, l = l, unit = "pt"))

  p
}


#'Internal helper function of viz_thickforest and viz_forest to create a thick forest plot
#'
#'Creates a thick forest plot. Called by viz_thickforest and viz_forest for type = "thick"
#'@keywords internal
internal_viz_thickforest <- function(plotdata, madata,
                                     type = "standard",
                                     study_labels = NULL, summary_label = NULL,
                                     study_table = NULL, summary_table = NULL, annotate_CI = FALSE,
                                     confidence_level = 0.95, col = "Blues", summary_col = "Blues", tick_col = "firebrick",
                                     text_size = 3, xlab = "Effect", x_limit = NULL,
                                     x_trans_function = NULL, x_breaks = NULL) {

  n <- nrow(plotdata)
  k <- length(levels(plotdata$group))

  # weight of each study used to scale the height of each raindrop
  if(type %in% c("standard", "study_only")) {
    weight <- 1/(plotdata$se^2 + madata$summary_tau2[as.numeric(plotdata$group)])
  } else {
    weight <- 1/plotdata$se^2
  }
  rel_weight <- weight/sum(weight)
  plotdata$rel_weight <- rel_weight
  plotdata <- plotdata %>%
    mutate(y_max = ID + rel_weight/(4*max(rel_weight)),
           y_min = ID - rel_weight/(4*max(rel_weight))
    )

  tick_size <- max(plotdata$rel_weight/(6*max(plotdata$rel_weight)))
  tickdata <- data.frame(x = c(plotdata$x, plotdata$x), ID = c(plotdata$ID, plotdata$ID),
                         y = c(plotdata$ID + tick_size,
                               plotdata$ID - tick_size))

  # set limits and breaks for the y axis and construct summary diamond (for type standard and sensitivity)
  if(type %in% c("standard", "sensitivity", "cumulative")) {
    y_limit <- c(min(plotdata$ID) - 3, max(plotdata$ID) + 1.5)
    y_tick_names <- c(as.vector(study_labels), as.vector(summary_label))[order(c(plotdata$ID, madata$ID), decreasing = T)]
    y_breaks <- sort(c(plotdata$ID, madata$ID), decreasing = T)
    summarydata <- data.frame("x.diamond" = c(madata$summary_es - stats::qnorm(1 - (1 - confidence_level) / 2, 0, 1) * madata$summary_se,
                                              madata$summary_es,
                                              madata$summary_es + stats::qnorm(1 - (1 - confidence_level) / 2, 0, 1) * madata$summary_se,
                                              madata$summary_es),
                              "y.diamond" = c(madata$ID,
                                              madata$ID + 0.3,
                                              madata$ID,
                                              madata$ID - 0.3),
                              "diamond_group" = rep(1:k, times = 4)
    )
  } else {
    y_limit <- c(min(plotdata$ID) - 1, max(plotdata$ID) + 1.5)
    y_tick_names <- plotdata$labels[order(plotdata$ID, decreasing = T)]
    y_breaks <- sort(plotdata$ID, decreasing = T)
  }


  # set limits for the x axis if none are supplied
  if(is.null(x_limit)) {
    x_limit <- c(range(c(plotdata$x_min, plotdata$x_max))[1] - diff(range(c(plotdata$x_min, plotdata$x_max)))*0.05,
                 range(c(plotdata$x_min, plotdata$x_max))[2] + diff(range(c(plotdata$x_min, plotdata$x_max)))*0.05)
  }

  # Set Color palette for shading
  if(type != "summary_only") {
    if(all(col %in% c("Blues", "Greys", "Oranges", "Greens", "Reds", "Purples"))) {
      col <- unlist(lapply(col, function(x) RColorBrewer::brewer.pal(n = 9, name = x)[9]))
    }
  }
  if(type != "study_only") {
    if(all(summary_col %in% c("Blues", "Greys", "Oranges", "Greens", "Reds", "Purples"))) {
      summary_col <- unlist(lapply(summary_col, function(x) RColorBrewer::brewer.pal(n = 9, name = x)[9]))
    }
    if(type == "summary_only") {
      col <- summary_col
    } else {
      if(length(summary_col) > 1) summary_col <- rep(summary_col, times = 4)
    }
  }



  # Set plot margins. If table is aligned on the left, no y axus breaks and ticks are plotted
  l <- 5.5
  r <- 11
  if(annotate_CI == TRUE) {
    r <- 1
  }
  if(!is.null(study_table) || !is.null(summary_table)) {
    l <- 1
    y_tick_names <- NULL
    y_breaks <- NULL
  }
  # workaround for "Undefined global functions or variables" Note in R CMD check while using ggplot2.
  x.diamond <- NULL
  y.diamond <- NULL
  diamond_group <- NULL
  x <- NULL
  y <- NULL
  x_min <- NULL
  x_max <- NULL
  y_min <- NULL
  y_max <- NULL
  ID <- NULL

  # Create thick forest plot
  p <-
    ggplot(data = plotdata, aes(y = ID, x = x)) +
    geom_errorbarh(data = plotdata, col = col, aes(xmin = x_min, xmax = x_max, y = ID, height = 0)) +
    geom_rect(aes(xmin = x_min, xmax = x_max, ymin = y_min, ymax = y_max,
                  group = ID), fill = col, size = 0.1) +
    geom_line(data = tickdata, aes(x = x, y = y, group = ID), col = tick_col, size = 1.5)
  if(type %in% c("standard", "sensitivity", "cumulative")) {
    p <- p + geom_polygon(data = summarydata, aes(x = x.diamond, y = y.diamond, group = diamond_group), color= "black", fill = summary_col, size = 0.1)
  }
  p <- p +
    geom_vline(xintercept = 0, linetype = 2) +
    scale_y_continuous(name = "",
                       breaks = y_breaks,
                       labels = y_tick_names) +
    coord_cartesian(xlim = x_limit, ylim = y_limit, expand = F)
  if(!is.null(x_trans_function)) {
    if(is.null(x_breaks)) {
      p <- p +
        scale_x_continuous(name = xlab,
                           labels = function(x) {round(x_trans_function(x), 3)})
    } else {
      p <- p +
        scale_x_continuous(name = xlab,
                           labels = function(x) {round(x_trans_function(x), 3)},
                           breaks = x_breaks)
    }
  } else {
    if(is.null(x_breaks)) {
      p <- p +
        scale_x_continuous(name = xlab)
    } else {
      p <- p +
        scale_x_continuous(breaks = x_breaks,
                           name = xlab)
    }
  }
  p <- p +
    theme_bw() +
    theme(text = element_text(size = 1/0.352777778*text_size),
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.x = element_line("grey"),
          panel.grid.minor.x = element_line("grey"),
          plot.margin = margin(t = 5.5, r = r, b = 5.5, l = l, unit = "pt"))
  p
}



#'Internal helper function of viz_forest to create a classic forest plot
#'
#'Creates a classic forest plot. Called by viz_forest for type = "classic"
#'@keywords internal
internal_viz_classicforest <- function(plotdata, madata,
                                       type = "standard",
                                       study_labels = NULL, summary_label = NULL,
                                       study_table = NULL, summary_table = NULL, annotate_CI = FALSE,
                                       confidence_level = 0.95, col = "Blues", summary_col = "Blues", tick_col = "firebrick",
                                       text_size = 3, xlab = "Effect", x_limit = NULL,
                                       x_trans_function = NULL, x_breaks = NULL) {
  n <- nrow(plotdata)
  k <- length(levels(plotdata$group))

  # weight of each study used to scale the height of each raindrop
  if(type %in% c("standard", "study_only")) {
    weight <- 1/(plotdata$se^2 + madata$summary_tau2[as.numeric(plotdata$group)])
  } else {
    weight <- 1/plotdata$se^2
  }
  plotdata$rel_weight <- weight/sum(weight)

  if(type %in% c("cumulative", "sensitivity")) {
    tick_size <- max(plotdata$rel_weight/(6*max(plotdata$rel_weight)))
    tickdata <- data.frame(x = c(plotdata$x, plotdata$x), ID = c(plotdata$ID, plotdata$ID),
                           y = c(plotdata$ID + tick_size,
                                 plotdata$ID - tick_size))
  }


  # set limits and breaks for the y axis and construct summary diamond (for type standard and sensitivity)
  if(type %in% c("standard", "sensitivity", "cumulative")) {
    y_limit <- c(min(plotdata$ID) - 3, max(plotdata$ID) + 1.5)
    y_tick_names <- c(as.vector(study_labels), as.vector(summary_label))[order(c(plotdata$ID, madata$ID), decreasing = T)]
    y_breaks <- sort(c(plotdata$ID, madata$ID), decreasing = T)
    summarydata <- data.frame("x.diamond" = c(madata$summary_es - stats::qnorm(1 - (1 - confidence_level) / 2, 0, 1) * madata$summary_se,
                                              madata$summary_es,
                                              madata$summary_es + stats::qnorm(1 - (1 - confidence_level) / 2, 0, 1) * madata$summary_se,
                                              madata$summary_es),
                              "y.diamond" = c(madata$ID,
                                              madata$ID + 0.3,
                                              madata$ID,
                                              madata$ID - 0.3),
                              "diamond_group" = rep(1:k, times = 4)
    )
  } else {
    y_limit <- c(min(plotdata$ID) - 1, max(plotdata$ID) + 1.5)
    y_tick_names <- plotdata$labels[order(plotdata$ID, decreasing = T)]
    y_breaks <- sort(plotdata$ID, decreasing = T)
  }

  # set limits for the x axis if none are supplied
  if(is.null(x_limit)) {
    x_limit <- c(range(c(plotdata$x_min, plotdata$x_max))[1] - diff(range(c(plotdata$x_min, plotdata$x_max)))*0.05,
                 range(c(plotdata$x_min, plotdata$x_max))[2] + diff(range(c(plotdata$x_min, plotdata$x_max)))*0.05)
  }

  # Set Color palette for shading
  if(type != "summary_only") {
    if(all(col %in% c("Blues", "Greys", "Oranges", "Greens", "Reds", "Purples"))) {
      col <- unlist(lapply(col, function(x) RColorBrewer::brewer.pal(n = 9, name = x)[9]))
    }
  }
  if(type != "study_only") {
    if(all(summary_col %in% c("Blues", "Greys", "Oranges", "Greens", "Reds", "Purples"))) {
      summary_col <- unlist(lapply(summary_col, function(x) RColorBrewer::brewer.pal(n = 9, name = x)[9]))
    }
    if(type == "summary_only") {
      col <- summary_col
    } else {
      if(length(summary_col) > 1) summary_col <- rep(summary_col, times = 4)
    }
  }


  # Set plot margins. If table is aligned on the left, no y axus breaks and ticks are plotted
  l <- 5.5
  r <- 11
  if(annotate_CI == TRUE) {
    r <- 1
  }
  if(!is.null(study_table) || !is.null(summary_table)) {
    l <- 1
    y_tick_names <- NULL
    y_breaks <- NULL
  }
  # workaround for "Undefined global functions or variables" Note in R CMD check while using ggplot2.
  x.diamond <- NULL
  y.diamond <- NULL
  diamond_group <- NULL
  ID <- NULL
  x <- NULL
  y <- NULL
  x_min <- NULL
  x_max <- NULL

  # create classic forest plot
  p <-
    ggplot(data = plotdata, aes(y = ID, x = x)) +
    geom_vline(xintercept = 0, linetype = 2) +
    geom_errorbarh(data = plotdata, col = "black", aes(xmin = x_min, xmax = x_max, y = ID, height = 0))

  if(type %in% c("cumulative", "sensitivity")) {
    p <- p + geom_line(data = tickdata, aes(x = x, y = y, group = ID), col = col, size = 1)
  } else {
    p <- p + geom_point(aes(size = weight), shape = 22, col = "black", fill = col)
  }

  if(type %in% c("standard", "sensitivity", "cumulative")) {
    p <- p + geom_polygon(data = summarydata, aes(x = x.diamond, y = y.diamond, group = diamond_group), color= "black", fill = summary_col, size = 0.1)
  }
  p <- p +
    scale_y_continuous(name = "",
                       breaks = y_breaks,
                       labels = y_tick_names) +
    coord_cartesian(xlim = x_limit, ylim = y_limit, expand = F)
  if(!is.null(x_trans_function)) {
    if(is.null(x_breaks)) {
      p <- p +
        scale_x_continuous(name = xlab,
                           labels = function(x) {round(x_trans_function(x), 3)})
    } else {
      p <- p +
        scale_x_continuous(name = xlab,
                           labels = function(x) {round(x_trans_function(x), 3)},
                           breaks = x_breaks)
    }
  } else {
    if(is.null(x_breaks)) {
      p <- p +
        scale_x_continuous(name = xlab)
    } else {
      p <- p +
        scale_x_continuous(breaks = x_breaks,
                           name = xlab)
    }
  }
  p <- p +
    scale_size_area(max_size = 3) +
    theme_bw() +
    theme(text = element_text(size = 1/0.352777778*text_size),
          legend.position = "none",
          panel.grid.major.y = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.grid.major.x = element_line("grey"),
          panel.grid.minor.x = element_line("grey"),
          plot.margin = margin(t = 5.5, r = r, b = 5.5, l = l, unit = "pt"))
  p
}


viz_forest_custom <- function(x, group = NULL, type = "standard", variant = "classic", method = "FE",
                            study_labels = NULL, summary_label = NULL,
                            confidence_level = 0.95, col = "Blues", summary_col = col,
                            text_size = 3, xlab = "Effect", x_limit = NULL,
                            x_trans_function = NULL, x_breaks = NULL,
                            annotate_CI = FALSE, study_table = NULL, summary_table = NULL,
                            table_headers = NULL, table_layout = NULL,
                            FDR = NULL, ...) {
  #'@import ggplot2
  #'@import dplyr


# Handle input object -----------------------------------------------------
  if(missing(x)) {
    stop("argument x is missing, with no default.")
  }
  # input is output of rma (metafor)
  if("rma" %in% class(x)) {
    es <- as.numeric(x$yi)
    se <- as.numeric(sqrt(x$vi))
    n <- length(es)

    # check if group argument has the right length
    if(!is.null(group) & (length(group) != length(es))) {
      warning("length of supplied group vector does not correspond to the number of studies; group argument is ignored.")
      group <- NULL
    }
    if(method != x$method) {
      method <- x$method
      # message("Note: method argument used differs from input object of class rma.uni (metafor)")
    }
    # If No group is supplied try to extract group from input object of class rma.uni (metafor)
    if(is.null(group) && ncol(x$X) > 1) {
      #check if only categorical moderators were used
      if(!all(x$X == 1 || x$X == 0) || any(apply(as.matrix(x$X[, -1]), 1, sum) > 1))  {
        stop("Can not deal with metafor output object with continuous and/or more than one categorical moderator variable(s).")
      }
      # extract group vector from the design matrix of the metafor object
      no.levels <- ncol(x$X) - 1
      group <- factor(apply(as.matrix(x$X[, -1])*rep(1:no.levels, each = n), 1, sum))
    }
  } else {
    # input is matrix or data.frame with effect sizes and standard errors in the first two columns
    if((is.data.frame(x) || is.matrix(x)) && ncol(x) >= 2) { # check if a data.frame or matrix with at least two columns is supplied
      # check if there are missing values
      if(sum(is.na(x[, 1])) != 0 || sum(is.na(x[, 2])) != 0) {
        warning("The effect sizes or standard errors contain missing values, only complete cases are used.")
        study_labels <- study_labels[stats::complete.cases(x[, c(1, 2)])]
        if(!is.null(group)) {
          group <- group[stats::complete.cases(x)]
        }
        x <- x[stats::complete.cases(x), ]
      }
      # check if input is numeric
      if(!is.numeric(x[, 1]) || !is.numeric(x[, 2])) {
        stop("Input argument has to be numeric; see help(viz_forest) for details.")
      }
      # check if there are any negative standard errors
      if(!all(x[, 2] >= 0)) {
        stop("Negative standard errors supplied")
      }
      # extract effects and standard errors
      es <- x[, 1]
      se <- x[, 2]
      n <- length(es)
    } else {
      stop("Unknown input argument. See help ('metaviz').")
    }
}

# Preprocess data ---------------------------------------------------------
  # check if group is a factor
  if(!is.null(group) && !is.factor(group)) {
    group <- as.factor(group)
  }
  # check if group vector has the right length
  if(!is.null(group) && (length(group) != length(es))) {
    warning("length of supplied group vector does not correspond to the number of studies; group argument is ignored")
    group <- NULL
  }

  # if no group argument is supplied, use all cases
  if(is.null(group)) {
    group <- factor(rep(1, times = n))
  }

  # drop unused levels of group factor
  group <- droplevels(group)
  k <- length(levels(group))

  # main data
  x <- data.frame(es, se, group)


  # check col is of length 1, or nrow(x) in case of variant classic or thick
  if(type != "summary_only") {
    if(variant == "rain") {
      stopifnot(length(col) == 1)
    } else {
      if(variant == "thick" || variant == "classic") {
        stopifnot(length(col) == 1 || length(col) == nrow(x))
      }
    }
  }
  # check summary_col is of length 1, or length(levels(group)) in case of variant classic or thick
  if(type != "study_only") {
    if(variant == "rain") {
      stopifnot(length(summary_col) == 1)
    } else {
      if(variant == "thick" || variant == "classic") {
        stopifnot(length(summary_col) == 1 || length(summary_col) == k)
      }
    }
  }

  if(n <= 1 && type == "sensitivity") {
    stop('For type = "sensitivity" there has to be more than 1 study.')
  }


  # Compute meta-analytic summary effect estimates
  if(type %in% c("standard", "summary_only", "sensitivity", "cumulative")) {
    M <- NULL # To avoid "no visible binding for global variable" warning for non-standard evaluation
    # compute meta-analytic summary effect for each group
    get_bse <- function(es, se, type = "b") {
      res <- metafor::rma.uni(yi = es, sei = se, method = method)
      if(type == "b") {
        res$b[[1]]
      } else {
        if(type == "se") {
          res$se[[1]]
        } else {
          stop()
        }
      }
    }
    M <- x %>%
      group_by(group) %>%
      summarise(M = get_bse(es, se, type = "b")) %>%
      select(M)
      summary_es <-  unlist(M)

    # compute standard error of the meta-analytic summary effect for each group
    M <- x %>%
      group_by(group) %>%
      summarise(M = get_bse(es, se, type = "se")) %>%
      select(M)
      summary_se <- unlist(M)

    if(type == "sensitivity") {
      loo <- function(es, se, type = "b") {
        res <- numeric(length(es))
        if(type == "b") {
          for(i in 1:length(es)) {
            res[i] <- metafor::rma.uni(yi = es[-i], sei = se[-i], method = method)$b[[1]]
          }
          res
        } else {
          if(type == "se") {
            for(i in 1:length(es)) {
              res[i] <- metafor::rma.uni(yi = es[-i], sei = se[-i], method = method)$se[[1]]
            }
            res
          } else {
            stop()
          }
        }
      }
      sens_data <- x %>%
        group_by(group) %>%
        mutate(summary_es = loo(es, se, type = "b"),
               summary_se = loo(es, se, type = "se"))
    }
    if(type == "cumulative") {
      rollingma <- function(es, se, type = "b") {
        res <- numeric(length(es))
        if(type == "b") {
          for(i in 1:length(es)) {
            res[i] <- metafor::rma.uni(yi = es[1:i], sei = se[1:i], method = method)$b[[1]]
          }
          res
        } else {
          if(type == "se") {
            for(i in 1:length(es)) {
              res[i] <- metafor::rma.uni(yi = es[1:i], sei = se[1:i], method = method)$se[[1]]
            }
            res
          } else {
            stop()
          }
        }
      }
      cum_data <- x %>%
        group_by(group) %>%
        mutate(summary_es = rollingma(es, se, type = "b"),
               summary_se = rollingma(es, se, type = "se"))
    }
  } else {
    if(type != "study_only") {
     stop('Argument of type must be one of "standard", "study_only", "summary_only", "cumulative", or "sensitivity".')
    }
  }

  # Compute tau^2 estimate
  if(type %in% c("standard", "study_only")) {
    if(method != "FE") {
    # compute tau squared for each group
    get_tau2 <- function(es, se) {
      metafor::rma.uni(yi = es, sei = se, method = method)$tau2[[1]]
    }
    M <- x %>%
      group_by(group) %>%
      summarise(M = get_tau2(es, se)) %>%
      select(M)
     summary_tau2 <- unlist(M)
    } else {
      summary_tau2 <- rep(0, times = k)
    }
  }

  if(type %in% c("study_only")) {
    if(!is.null(summary_table) || !is.null(summary_label)) {
      warning('For type "study_only" supplied summary_table and summary_label are ignored.')
    }
    summary_table <- NULL
    summary_label <- NULL
  }
  if(type == "summary_only") {
    if(!is.null(study_table) || !is.null(study_labels)) {
      warning('For type "summary_only" supplied study_table and study_labels are ignored.')
    }
    study_table <- NULL
    study_labels <- NULL
  }


  # if not exactly one name for every study is supplied the default is used (numbers 1 to the number of studies)
  if(is.null(study_labels) || length(study_labels) != n) {
    if(!is.null(study_labels) && length(study_labels) != n) {
      warning("Argument study_labels has wrong length and is ignored.")
    }
    study_labels <- 1:n
  }

  # if not exactly one name for every subgroup is suppied the default is used
  if(is.null(summary_label) || length(summary_label) != k) {
    if(!is.null(summary_label) && length(summary_label) != k) {
      warning("Argument summary_label has wrong length and is ignored.")
    }
    if(k != 1) {
      summary_label <- paste("Subgroup: ", levels(group), sep = "")
    } else {
      summary_label <- "Summary"
    }
  }

  if(confidence_level <= 0 || confidence_level >= 1) {
    stop("Argument confidence_level must be larger than 0 and smaller than 1.")
  }

  if(!is.null(x_trans_function) && !is.function(x_trans_function)) {
    warning("Argument x_trans_function must be a function; input ignored.")
    x_trans_function <- NULL
  }

  if(!is.null(table_layout) && !is.matrix(table_layout)) {
    warning("Agument of table_layout is not a matrix and is ignored.")
    table_layout <- NULL
  }

  # Determine IDs for studies and summary effects which correspond to plotting y coordinates
  ids <- function(group, n) {
    k <- length(levels(group))
    ki_start <- cumsum(c(1, as.numeric(table(group))[-k] + 3))
    ki_end <- ki_start + as.numeric(table(group)) - 1
    study_IDs <- numeric(n)
    for(i in 1:k) {
      study_IDs[group==levels(group)[i]] <- ki_start[i]:ki_end[i]
    }
    summary_IDs <- ki_end + 2
    data.frame("ID" = (n + 3*k-2) - c(study_IDs, summary_IDs),
               "type" = factor(c(rep("study", times = length(study_IDs)),
                                 rep ("summary", times = length(summary_IDs)))))
  }
  if(type != "summary_only") {
  ID <- ids(group, n = n)
  } else {
    ID <- ids(unique(group), n = k)
  }

  if(type %in% c("standard", "study_only")) {
    plotdata <- data.frame("x" = es, "se" = se,
                           "ID" = ID$ID[ID$type == "study"],
                           "labels" = study_labels,
                           "group"= group,
                           "x_min" = es - stats::qnorm(1 - (1 - confidence_level)/2)*se,
                           "x_max" = es + stats::qnorm(1 - (1 - confidence_level)/2)*se)

    if(type == "standard") {
      madata <- data.frame("summary_es" = summary_es,
                           "summary_se" = summary_se,
                           "summary_tau2" = summary_tau2,
                           "ID" = ID$ID[ID$type == "summary"])
    }
    if(type == "study_only") {
      madata <- data.frame("summary_tau2" = summary_tau2)
    }
  } else {
    if(type == "summary_only") {
      plotdata <- data.frame("x" = summary_es, "se" = summary_se,
                             "ID" = ID$ID[ID$type == "summary"],
                             "labels" = summary_label,
                             "group"= levels(group),
                             "x_min" = summary_es - stats::qnorm(1 - (1 - confidence_level)/2)*summary_se,
                             "x_max" = summary_es + stats::qnorm(1 - (1 - confidence_level)/2)*summary_se)
      madata <- NULL
    } else {
      if(type == "cumulative") {
        plotdata <- data.frame("x" = cum_data$summary_es, "se" = cum_data$summary_se,
                               "ID" = ID$ID[ID$type == "study"],
                               "labels" = study_labels,
                               "group"= group,
                               "x_min" = cum_data$summary_es - stats::qnorm(1 - (1 - confidence_level)/2)*cum_data$summary_se,
                               "x_max" = cum_data$summary_es + stats::qnorm(1 - (1 - confidence_level)/2)*cum_data$summary_se)
        madata <- data.frame("summary_es" = summary_es,
                             "summary_se" = summary_se,
                             "ID" = ID$ID[ID$type == "summary"])
      } else {
        if(type == "sensitivity") {
          plotdata <- data.frame("x" = sens_data$summary_es, "se" = sens_data$summary_se,
                                 "ID" = ID$ID[ID$type == "study"],
                                 "labels" = study_labels,
                                 "group"= group,
                                 "x_min" = sens_data$summary_es - stats::qnorm(1 - (1 - confidence_level)/2)*sens_data$summary_se,
                                 "x_max" = sens_data$summary_es + stats::qnorm(1 - (1 - confidence_level)/2)*sens_data$summary_se)
          madata <- data.frame("summary_es" = summary_es,
                               "summary_se" = summary_se,
                               "ID" = ID$ID[ID$type == "summary"])
        }
      }
    }
  }

# Create forest plot variant ------------------------------------------------------
  args <- c(list(plotdata = plotdata, madata = madata,
            type = type,
            study_labels = study_labels, summary_label = summary_label,
            study_table = study_table, summary_table = summary_table,
            annotate_CI = annotate_CI, confidence_level = confidence_level, col = col,
            summary_col = summary_col,
            text_size = text_size, xlab = xlab, x_limit = x_limit,
            x_trans_function = x_trans_function, x_breaks = x_breaks), list(...))

  if(variant == "rain") {
    p <- do.call(internal_viz_rainforest, args)
  } else {
    if(variant == "thick") {
      p <- do.call(internal_viz_thickforest, args)
    } else {
      if(variant == "classic") {
        p <- do.call(internal_viz_classicforest, args)
      } else {
        stop("The argument of variant must be one of rain, thick or classic.")
      }
    }
  }

# Construct tableplots with study and summary information --------
  if(annotate_CI == TRUE || !is.null(study_table) || !is.null(summary_table)) {

    # set limits for the y axis of the table plots
    if(type %in% c("standard", "sensitivity", "cumulative")) {
      y_limit <- c(min(plotdata$ID) - 3, max(plotdata$ID) + 1.5)
    } else {
      y_limit <- c(min(plotdata$ID) - 1, max(plotdata$ID) + 1.5)
    }

    # Function to create table plots
    table_plot <- function(tbl, ID, r = 5.5, l = 5.5, tbl_titles = NULL) {
      # all columns and column names are stacked to a vector
      df_to_vector <- function(df) {
        v <- vector("character", 0)
        for(i in 1:ncol(df)) v <- c(v, as.vector(df[, i]))
        v
      }
      if(!is.data.frame(tbl)) tbl <- data.frame(tbl)
      tbl <- data.frame(lapply(tbl, as.character), stringsAsFactors = FALSE)
      if(is.null(tbl_titles)) {
        tbl_titles <- names(tbl)
      }
      v <- df_to_vector(tbl)

      # For study labels with newlines in it, the width of the column is now set according to longest line and not the whole label
      nchar2<-function(x){unlist(sapply(strsplit(x,"\n"), function(x) max(nchar(x, keepNA = FALSE))))}
      area_per_column <- cumsum(c(1, apply(rbind(tbl_titles, tbl), 2, function(x) max(round(max(nchar2(x))/100, 2),  0.03))))
      #area_per_column <- cumsum(c(1, apply(rbind(tbl_titles, tbl), 2, function(x) max(round(max(nchar(x, keepNA = FALSE))/100, 2),  0.03))))

      x_values <- area_per_column[1:ncol(tbl)]
      x_limit <- range(area_per_column)

      lab <- data.frame(y = rep(ID, ncol(tbl)),
                        x = rep(x_values,
                                each = length(ID)),
                        value = v, stringsAsFactors = FALSE)

      lab_title <- data.frame(y = rep(max(plotdata$ID) + 1, times = length(tbl_titles)),
                              x = x_values,
                              value = tbl_titles)

      # To avoid "no visible binding for global variable" warning for non-standard evaluation
      y <- NULL
      value <- NULL
      ggplot(lab, aes(x = x, y = y)) +
        geom_text(aes(label = value), size = text_size, hjust = 0, vjust = 0.5) +
        geom_text(data = lab_title, aes(x = x, y = y, label = value), size = text_size, hjust = 0, vjust = 0.5) +
        coord_cartesian(xlim = x_limit, ylim = y_limit, expand = F) +
        geom_hline(yintercept = max(plotdata$ID) + 0.5) +
        theme_bw() +
        theme(text = element_text(size = 1/0.352777778*text_size),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              legend.position = "none",
              panel.border = element_blank(),
              axis.text.x = element_text(colour="white"),
              axis.text.y = element_blank(),
              axis.ticks.x = element_line(colour="white"),
              axis.ticks.y = element_blank(),
              axis.line.x = element_line(colour="white"),
              axis.line.y = element_blank(),
              plot.margin = margin(t = 5.5, r = r, b = 5.5, l = l, unit = "pt")) +
        labs(x = "", y = "")
    }

    # Study and/or summary table left
    if(!is.null(study_table) || !is.null(summary_table)) {
      # Case study table and summary table are both supplied (type standard, cumulative, or sensitivity)
      if(!is.null(study_table) && !is.null(summary_table)) {
        if(!is.data.frame(study_table)) study_table <- data.frame(study_table)
        if(!is.data.frame(summary_table)) summary_table <- data.frame(summary_table)
        study_table <- data.frame(lapply(study_table, as.character), stringsAsFactors = FALSE)
        summary_table <- data.frame(lapply(summary_table, as.character), stringsAsFactors = FALSE)
        if(nrow(study_table) != n) stop('study_table must be a data.frame with one row for each study.')
        if(nrow(summary_table) != k) stop('summary_table must be a data.frame with one row for each summary effect.')
        if(ncol(summary_table) < ncol(study_table)) {
          n_fillcol <- ncol(study_table) - ncol(summary_table)
          summary_table <- data.frame(summary_table, matrix(rep("", times = nrow(summary_table) * n_fillcol), ncol = n_fillcol))
          summary_table<- stats::setNames(summary_table, names(study_table))
        } else {
          if(ncol(summary_table) > ncol(study_table)) {
            n_fillcol <- ncol(summary_table) - ncol(study_table)
            study_table <- data.frame(study_table, matrix(rep("", times = nrow(study_table) * n_fillcol), ncol = n_fillcol))
            study_table <- stats::setNames(study_table, names(summary_table))
          }
        }
        if(any(names(study_table) != names(summary_table))) summary_table <- stats::setNames(summary_table, names(study_table))
      } else {
        # Case only study table is supplied
        if(is.null(summary_table)) {
          if(type %in% c("standard", "sensitivity", "cumulative", "study_only")) {
            if(!is.data.frame(study_table)) study_table <- data.frame(study_table)
            study_table <- data.frame(lapply(study_table, as.character), stringsAsFactors = FALSE)
            if(nrow(study_table) != n) stop('study_table must be a data.frame with one row for each study.')
            summary_table <- as.data.frame(matrix(rep("", times = ncol(study_table) * k), ncol = ncol(study_table)), stringsAsFactors = FALSE)
            summary_table <- stats::setNames(summary_table, names(study_table))
          }
        }
        # Case only summary table is supplied
        if(is.null(study_table)) {
          if(type %in% c("standard", "sensitivity", "cumulative")) {
            if(!is.data.frame(summary_table)) summary_table <- data.frame(summary_table)
            summary_table <- data.frame(lapply(summary_table, as.character), stringsAsFactors = FALSE)
            if(nrow(summary_table) != k) stop('summary_table must be a data.frame with one row for each summary effect.')
            study_table <- as.data.frame(matrix(rep("", times = ncol(summary_table) * n), ncol = ncol(summary_table)), stringsAsFactors = FALSE)
            study_table <- stats::setNames(study_table, names(summary_table))
          } else {
            if(type %in% c("summary_only")) {
              if(!is.data.frame(summary_table)) summary_table <- data.frame(summary_table)
              summary_table <- data.frame(lapply(summary_table, as.character), stringsAsFactors = FALSE)
              if(nrow(summary_table) != k) stop('summary_table must be a data.frame with one row for each summary effect.')
              study_table <- as.data.frame(matrix(rep("", times = ncol(summary_table) * k), ncol = ncol(summary_table)), stringsAsFactors = FALSE)
              study_table <- stats::setNames(study_table, names(summary_table))
            }
          }
        }
      }

    table_left <- data.frame(rbind(study_table, summary_table))


    # set table headers
    if(!is.null(table_headers)) {
      if(length(table_headers) >= ncol(table_left)) {
        table_headers_left <- table_headers[1:ncol(table_left)]
      } else {
        warning("Argument table_headers has not the right length and is ignored.")
        table_headers_left <- NULL
      }
    } else {
      table_headers_left <- NULL
    }

  table_left_plot <- table_plot(table_left, ID = ID$ID, r = 0, tbl_titles = table_headers_left)
  } else {
    table_left <- NULL
  }

  # Textual CI and effect size values right
  if(annotate_CI == TRUE) {

    # set table headers
    if(!is.null(table_headers)) {
      if(is.null(table_left)) {
        if(length(table_headers) == 1) {
          table_headers_right <- table_headers
        } else {
          warning("Argument table_headers has not the right length and is ignored.")
          table_headers_right <- NULL
        }
      } else {
        if(length(table_headers) == ncol(table_left) + 1) {
          table_headers_right <- table_headers[ncol(table_left) + 1]
        } else {
          table_headers_right <- NULL
        }
      }
    } else {
      table_headers_right <- NULL
    }

    if(is.null(table_headers_right)){
      table_headers_right <- paste(xlab, " [", confidence_level*100, "% CI]", sep = "")
    }

    if(type %in% c("standard", "sensitivity", "cumulative")) {
      x_hat <- c(plotdata$x, madata$summary_es)
      lb <- c(c(plotdata$x, madata$summary_es) - stats::qnorm(1 - (1 - confidence_level)/2, 0, 1)*c(plotdata$se, madata$summary_se))
      ub <-  c(c(plotdata$x, madata$summary_es) + stats::qnorm(1 - (1 - confidence_level)/2, 0, 1)*c(plotdata$se, madata$summary_se))

      if(!is.null(x_trans_function)) {
        x_hat <- x_trans_function(x_hat)
        lb <- x_trans_function(lb)
        ub <- x_trans_function(ub)
      }

      lb <- format(round(lb, 2), nsmall = 2)
      ub <- format(round(ub, 2), nsmall = 2)
      x_hat <- format(round(x_hat, 2), nsmall = 2)

      CI <- paste(x_hat, " [", lb, ", ", ub, "]", sep = "")
      CI_label <- data.frame(CI = CI, stringsAsFactors = FALSE)

      table_CI <- table_plot(CI_label, ID = c(plotdata$ID, madata$ID), l = 0, r = 11,  tbl_titles = table_headers_right)
    } else {
      if(type %in% c("study_only", "summary_only")) {
        x_hat <- plotdata$x
        lb <- plotdata$x - stats::qnorm(1 - (1 - confidence_level)/2, 0, 1)*plotdata$se
        ub <-  plotdata$x + stats::qnorm(1 - (1 - confidence_level)/2, 0, 1)*plotdata$se

        if(!is.null(x_trans_function)) {
          x_hat <- x_trans_function(x_hat)
          lb <- x_trans_function(lb)
          ub <- x_trans_function(ub)
        }

        lb <- format(round(lb, 2), nsmall = 2)
        ub <- format(round(ub, 2), nsmall = 2)
        x_hat <- format(round(x_hat, 2), nsmall = 2)
        CI <- paste(x_hat, " [", lb, ", ", ub, "]", sep = "")
        CI_label <- data.frame(CI = CI, FDR = FDR, stringsAsFactors = FALSE)
        table_CI <- table_plot(CI_label, ID = plotdata$ID, l = 0, r = 11, tbl_titles = c(table_headers_right, 'FDR'))
      }
    }
  } else {
    table_CI <- NULL
  }
# Align forest plot and table(s) -----------------------------------
    if(!is.null(table_CI) && !is.null(table_left)) {
      if(is.null(table_layout)) {
        layout_matrix <- matrix(c(rep(1, times = ncol(table_left)), rep(2, times = 3), 3), nrow = 1)
      } else {
        layout_matrix <- table_layout
      }
      p <- gridExtra::arrangeGrob(table_left_plot, p, table_CI, layout_matrix = layout_matrix)
      ggpubr::as_ggplot(p)
    } else {
      if(!is.null(table_CI) && is.null(table_left)) {
        if(is.null(table_layout)) {
          layout_matrix <- matrix(c(1, 1, 1, 1, 2), nrow = 1)
        } else {
          layout_matrix <- table_layout
        }
        p <- gridExtra::arrangeGrob(p, table_CI, layout_matrix = layout_matrix)
        ggpubr::as_ggplot(p)
      } else {
        if(is.null(table_CI) && !is.null(table_left)) {
          if(is.null(table_layout)) {
            layout_matrix <- matrix(c(rep(1, times = 1 + ncol(table_left)), 2, 2, 2, 2, 2), nrow = 1)
          } else {
            layout_matrix <- table_layout
          }
          p <- gridExtra::arrangeGrob(table_left_plot, p, layout_matrix = layout_matrix)
          ggpubr::as_ggplot(p)
        }
      }
    }
  } else {
    p
  }
  # p
  # return(p)
}

library(mgcv)
library(sas7bdat)
library(tidyr)
library(metaviz)
library(ggplot2)
cex <- 0.1
par(cex.lab=cex, cex.axis=cex, cex.main=cex)

main <- read.csv ("r_summary_single_pollutant.csv", header = TRUE)
# dev.off()
# graphics.off()
fig2_df <- data.frame(
  Outcomes = main[,"out_first_only"],
  'Air toxics' =main$pol1
)
colnames(fig2_df) <- gsub("\\.", " ", colnames(fig2_df))

fig2 <- viz_forest_custom(x = main[, c("mean", "se")],x_trans_function = exp,x_limit =c(-0.5,2.2) ,
                        group=main[,"outcome"],
                        study_labels=main[,"outcome"],  col = "Greys",
                        x_breaks = c(0,0.693147,1.386294, 2.0794415), text_size = 2.3,
                        annotate_CI = TRUE,
                        xlab="OR",
                        study_table = fig2_df,
                        FDR = main$fdr,
                        type = "study_only",
                        table_layout = matrix(c(1,1,1,1, 2, 2, 3,3), nrow = 1))
pdf("20210909_Fig2_with_sample_num.pdf",width=7,height=5)

print(fig2)
# k2
dev.off()

main <- read.csv ("r_summary_all_greater.csv", header = TRUE)
main[main == ''] <- NA
# dev.off()
# graphics.off()
# dev.new()


pollutants_combinations <- apply(main[, c('pol1', 'pol2', 'pol3')], 1, function(x) paste(na.omit(x),collapse=" & ") )
fig3_df <- data.frame(
  Outcomes = main[,"out_first_only"],
  'Air toxics' = pollutants_combinations
)
colnames(fig3_df) <- gsub("\\.", " ", colnames(fig3_df))


fig3 <- viz_forest_custom(x = main[, c("mean", "se")],x_trans_function = exp,x_limit =c(-0.5,2.2) ,
                        group=main[,"outcome"],
                        study_labels=main[,"outcome"], x_breaks = c(0,0.693147,1.386294, 2.0794415), col = "Greys",  text_size = 2.28,
                        annotate_CI = TRUE,xlab="OR",
                        study_table = fig3_df,
                        FDR = main$fdr,
                        # summary_table = summary_table,
                        type = "study_only",
                        table_layout = matrix(c(1,1,1,1,1, 2, 2, 3,3), nrow = 1)) + geom_point()

pdf("20210909_Fig3_with_sample_num.pdf",width=7,height=5)
##w16 H10
print(fig3)
dev.off()


