install.packages("Rcpp")
install.packages("colorspace")
install.packages("plyr")
install.packages("scales")
install.packages("lazyeval")
install.packages("rlang")
install.packages("tibble")
install.packages("ggplot2")
install.packages("stringi")
install.packages("reshape2")
install.packages("digest")

library(ggplot2)
library(doBy)
library(plyr)
library(gridExtra)
ADNI_majvote = read.csv("ADNI_majvote.csv")
ADNI_majvote$NumAtlas = as.factor(ADNI_majvote$NumAtlas)
ADNI_majvote$NumTemplate = as.factor(ADNI_majvote$NumTemplate)
ADNI_PUBMRF = read.csv("ADNI_PUB-MRF.csv")
ADNI_PUBMRF$NumAtlas = as.factor(ADNI_PUBMRF$NumAtlas)
ADNI_PUBMRF$NumTemplate = as.factor(ADNI_PUBMRF$NumTemplate)
ADNI_JLF = read.csv("ADNI_JLF_default.csv")
ADNI_JLF$NumAtlas = as.factor(ADNI_JLF$NumAtlas)

se <- function(x) sd(x)/sqrt(length(x))
scaleFUN <- function(x) sprintf("%.3f", x)

MinMeanSEMMax <- function(x) {
  v <- c(mean(x) - 2*sd(x), mean(x) - 2*sd(x)/sqrt(length(x)), mean(x), mean(x) + 2*sd(x)/sqrt(length(x)), mean(x) + 2*sd(x))
  names(v) <- c("ymin", "lower", "middle", "upper", "ymax")
  v
}


#ADNI, hippocampus

ADNI_majvote$Type <- "Majority Vote"
ADNI_PUBMRF$Type <- "PUB-MRF"
ADNI_JLF$Type <- "JLF"
ADNI_plot1 <- rbind(ADNI_majvote, ADNI_PUBMRF)
ADNI_plot1 = ADNI_plot1[ADNI_plot1$Label == "All",]

ADNI_majvote2 = ADNI_majvote[ADNI_majvote$NumAtlas == "9",]
ADNI_majvote2 = ADNI_majvote2[ADNI_majvote2$NumTemplate == "19",]
ADNI_PUBMRF2 = ADNI_PUBMRF[ADNI_PUBMRF$NumAtlas == "9",]
ADNI_PUBMRF2 = ADNI_PUBMRF2[ADNI_PUBMRF2$NumTemplate == "19",]
ADNI_JLF2 = ADNI_JLF[ADNI_JLF$NumAtlas == "9",]
ADNI_plot2 <- rbind.fill(ADNI_majvote2, ADNI_PUBMRF2, ADNI_JLF2)
ADNI_plot2 = ADNI_plot2[ADNI_plot2$Label == "All",]
ADNI_plot2$Type <- factor(ADNI_plot2$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=ADNI_plot1, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.780, 0.880, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(ADNI_plot2, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.780, 0.960, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

mean(ADNI_majvote2$Dice)
mean(ADNI_PUBMRF2$Dice)
mean(ADNI_JLF2$Dice)

majvote_vs_PUBMRF = t.test(ADNI_majvote2$Dice, ADNI_PUBMRF2$Dice)
majvote_vs_JLF = t.test(ADNI_majvote2$Dice, ADNI_JLF2$Dice)
PUBMRF_vs_JLF = t.test(ADNI_PUBMRF2$Dice, ADNI_JLF2$Dice)

df1 <- data.frame(a = c(1, 1:3, 3), b = c(0.953, 0.955, 0.955, 0.955, 0.953))
df2 <- data.frame(a = c(2, 2,3, 3), b = c(0.946, 0.948, 0.948, 0.946))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2, y = 0.955, label = "**", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.948, label = "**", size = 8)


FEP_majvote = read.csv("FEP_majvote.csv")
FEP_majvote$NumAtlas = as.factor(FEP_majvote$NumAtlas)
FEP_majvote$NumTemplate = as.factor(FEP_majvote$NumTemplate)
FEP_PUBMRF = read.csv("FEP_PUB-MRF.csv")
FEP_PUBMRF$NumAtlas = as.factor(FEP_PUBMRF$NumAtlas)
FEP_PUBMRF$NumTemplate = as.factor(FEP_PUBMRF$NumTemplate)
FEP_JLF = read.csv("FEP_JLF_default.csv")
FEP_JLF$NumAtlas = as.factor(FEP_JLF$NumAtlas)

se <- function(x) sd(x)/sqrt(length(x))
scaleFUN <- function(x) sprintf("%.3f", x)

# FEP, striatum

FEP_majvote$Type <- "Majority Vote"
FEP_PUBMRF$Type <- "PUB-MRF"
FEP_JLF$Type <- "JLF"
FEP_plot1 <- rbind(FEP_majvote, FEP_PUBMRF)
FEP_plot1_striatum <- FEP_plot1[FEP_plot1$Label %in% c('1', '4'),]

FEP_majvote2 = FEP_majvote[FEP_majvote$NumAtlas == "9",]
FEP_majvote2 = FEP_majvote2[FEP_majvote2$NumTemplate == "19",]
FEP_PUBMRF2 = FEP_PUBMRF[FEP_PUBMRF$NumAtlas == "9",]
FEP_PUBMRF2 = FEP_PUBMRF2[FEP_PUBMRF2$NumTemplate == "19",]
FEP_JLF2 = FEP_JLF[FEP_JLF$NumAtlas == "9",]
FEP_plot2 <- rbind.fill(FEP_majvote2, FEP_PUBMRF2, FEP_JLF2)
FEP_plot2_striatum <- FEP_plot2[FEP_plot2$Label %in% c('1', '4'),]
FEP_plot2_striatum$Type <- factor(FEP_plot2_striatum$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=FEP_plot1_striatum, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.890, 0.930, 0.010), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(FEP_plot2_striatum, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.910, 0.950, 0.010), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

FEP_majvote_st = FEP_plot2_striatum[FEP_plot2_striatum$Type == "Majority Vote",]
FEP_PUBMRF_st = FEP_plot2_striatum[FEP_plot2_striatum$Type == "PUB-MRF",]
FEP_JLF_st = FEP_plot2_striatum[FEP_plot2_striatum$Type == "JLF",]

mean(FEP_majvote_st$Dice)
mean(FEP_PUBMRF_st$Dice)
mean(FEP_JLF_st$Dice)

majvote_vs_PUBMRF = t.test(FEP_majvote_st$Dice, FEP_PUBMRF_st$Dice)
majvote_vs_JLF = t.test(FEP_majvote_st$Dice, FEP_JLF_st$Dice)
PUBMRF_vs_JLF = t.test(FEP_PUBMRF_st$Dice, FEP_JLF_st$Dice)

df1 <- data.frame(a = c(1, 1:3, 3), b = c(0.949, 0.950, 0.950, 0.950, 0.949))
df2 <- data.frame(a = c(2, 2,3, 3), b = c(0.946, 0.947, 0.947, 0.946))
df3 <- data.frame(a = c(1, 1,2, 2), b = c(0.943, 0.944, 0.944, 0.943))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2, y = 0.9505, label = "**", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.9475, label = "**", size = 8) +
  geom_line(data = df3, aes(x = a, y = b)) + annotate("text", x = 1.5, y = 0.9445, label = "*", size = 8)

# FEP, globus pallidus

FEP_plot1_GP <- FEP_plot1[FEP_plot1$Label %in% c('2', '5'),]
FEP_plot2_GP <- FEP_plot2[FEP_plot2$Label %in% c('2', '5'),]
FEP_plot2_GP$Type <- factor(FEP_plot2_GP$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=FEP_plot1_GP, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.750, 0.850, 0.010), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(FEP_plot2_GP, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.700, 0.960, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

FEP_majvote_GP = FEP_plot2_GP[FEP_plot2_GP$Type == "Majority Vote",]
FEP_PUBMRF_GP = FEP_plot2_GP[FEP_plot2_GP$Type == "PUB-MRF",]
FEP_JLF_GP = FEP_plot2_GP[FEP_plot2_GP$Type == "JLF",]

mean(FEP_majvote_GP$Dice)
mean(FEP_PUBMRF_GP$Dice)
mean(FEP_JLF_GP$Dice)

majvote_vs_PUBMRF = t.test(FEP_majvote_GP$Dice, FEP_PUBMRF_GP$Dice)
majvote_vs_JLF = t.test(FEP_majvote_GP$Dice, FEP_JLF_GP$Dice)
PUBMRF_vs_JLF = t.test(FEP_PUBMRF_GP$Dice, FEP_JLF_GP$Dice)

df1 <- data.frame(a = c(2, 2,3, 3), b = c(0.917, 0.919, 0.919, 0.917))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.9195, label = "*", size = 8)

# FEP, thalamus

FEP_plot1_th <- FEP_plot1[FEP_plot1$Label %in% c('3', '6'),]
FEP_plot2_th <- FEP_plot2[FEP_plot2$Label %in% c('3', '6'),]
FEP_plot2_th$Type <- factor(FEP_plot2_th$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=FEP_plot1_th, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.850, 0.910, 0.010), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(FEP_plot2_th, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.800, 0.980, 0.010), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

FEP_majvote_th = FEP_plot2_th[FEP_plot2_th$Type == "Majority Vote",]
FEP_PUBMRF_th = FEP_plot2_th[FEP_plot2_th$Type == "PUB-MRF",]
FEP_JLF_th = FEP_plot2_th[FEP_plot2_th$Type == "JLF",]

mean(FEP_majvote_th$Dice)
mean(FEP_PUBMRF_th$Dice)
mean(FEP_JLF_th$Dice)

majvote_vs_PUBMRF = t.test(FEP_majvote_th$Dice, FEP_PUBMRF_th$Dice)
majvote_vs_JLF = t.test(FEP_majvote_th$Dice, FEP_JLF_th$Dice)
PUBMRF_vs_JLF = t.test(FEP_PUBMRF_th$Dice, FEP_JLF_th$Dice)

df1 <- data.frame(a = c(2, 2,3, 3), b = c(0.952, 0.9535, 0.9535, 0.952))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.954, label = "*", size = 8)


HA_majvote = read.csv("HA_majvote.csv")
HA_majvote$NumAtlas = as.factor(HA_majvote$NumAtlas)
HA_majvote$NumTemplate = as.factor(HA_majvote$NumTemplate)
HA_PUBMRF = read.csv("HA_PUBMRF.csv")
HA_PUBMRF$NumAtlas = as.factor(HA_PUBMRF$NumAtlas)
HA_PUBMRF$NumTemplate = as.factor(HA_PUBMRF$NumTemplate)
HA_JLF = read.csv("HA_JLF.csv")
HA_JLF$NumAtlas = as.factor(HA_JLF$NumAtlas)

se <- function(x) sd(x)/sqrt(length(x))
scaleFUN <- function(x) sprintf("%.3f", x)

# HA, CA1

HA_majvote$Type <- "Majority Vote"
HA_PUBMRF$Type <- "PUB-MRF"
HA_JLF$Type <- "JLF"
HA_plot1 <- rbind(HA_majvote, HA_PUBMRF)
HA_plot1_CA1 <- HA_plot1[HA_plot1$Label %in% c('1', '101'),]

HA_majvote2 = HA_majvote[HA_majvote$NumAtlas == "9",]
HA_majvote2 = HA_majvote2[HA_majvote2$NumTemplate == "19",]
HA_PUBMRF2 = HA_PUBMRF[HA_PUBMRF$NumAtlas == "9",]
HA_PUBMRF2 = HA_PUBMRF2[HA_PUBMRF2$NumTemplate == "19",]
HA_JLF2 = HA_JLF[HA_JLF$NumAtlas == "9",]
HA_plot2 <- rbind.fill(HA_majvote2, HA_PUBMRF2, HA_JLF2)
HA_plot2_CA1 <- HA_plot2[HA_plot2$Label %in% c('1', '101'),]
HA_plot2_CA1$Type <- factor(HA_plot2_CA1$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice <- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=HA_plot1_CA1, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.500, 0.900, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

ggplot(HA_plot2_CA1, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.400, 0.900, 0.040), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

HA_majvote_CA1 = HA_plot2_CA1[HA_plot2_CA1$Type == "Majority Vote",]
HA_PUBMRF_CA1 = HA_plot2_CA1[HA_plot2_CA1$Type == "PUB-MRF",]
HA_JLF_CA1 = HA_plot2_CA1[HA_plot2_CA1$Type == "JLF",]

HA_majvote_CA1_test <- HA_majvote_CA1[-(1:20), ]
HA_PUBMRF_CA1_test <- HA_PUBMRF_CA1[-(1:20), ]
HA_JLF_CA1_test <- HA_JLF_CA1[-(1:20), ]

mean(HA_majvote_CA1$Dice)
mean(HA_PUBMRF_CA1$Dice)
mean(HA_JLF_CA1$Dice)

mean(HA_majvote_CA1_test$Dice)
mean(HA_PUBMRF_CA1_test$Dice)
mean(HA_JLF_CA1_test$Dice)

majvote_vs_PUBMRF = t.test(HA_majvote_CA1$Dice, HA_PUBMRF_CA1$Dice)
majvote_vs_JLF = t.test(HA_majvote_CA1$Dice, HA_JLF_CA1$Dice)
PUBMRF_vs_JLF = t.test(HA_PUBMRF_CA1$Dice, HA_JLF_CA1$Dice)

df1 <- data.frame(a = c(1, 1:3, 3), b = c(0.949, 0.950, 0.950, 0.950, 0.949))
df2 <- data.frame(a = c(2, 2,3, 3), b = c(0.946, 0.947, 0.947, 0.946))
df3 <- data.frame(a = c(1, 1,2, 2), b = c(0.943, 0.944, 0.944, 0.943))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2, y = 0.9505, label = "**", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.9475, label = "**", size = 8) +
  geom_line(data = df3, aes(x = a, y = b)) + annotate("text", x = 1.5, y = 0.9445, label = "*", size = 8)

# HA, subiculum

HA_plot1_sub <- HA_plot1[HA_plot1$Label %in% c('2', '102'),]
HA_plot2_sub <- HA_plot2[HA_plot2$Label %in% c('2', '102'),]
HA_plot2_sub$Type <- factor(HA_plot2_sub$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=HA_plot1_sub, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.380, 0.500, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp = ggplot(HA_plot2_sub, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.200, 0.900, 0.040), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

HA_majvote_sub = HA_plot2_sub[HA_plot2_sub$Type == "Majority Vote",]
HA_PUBMRF_sub = HA_plot2_sub[HA_plot2_sub$Type == "PUB-MRF",]
HA_JLF_sub = HA_plot2_sub[HA_plot2_sub$Type == "JLF",]

HA_majvote_sub_test <- HA_majvote_sub[-(1:20), ]
HA_PUBMRF_sub_test <- HA_PUBMRF_sub[-(1:20), ]
HA_JLF_sub_test <- HA_JLF_sub[-(1:20), ]

mean(HA_majvote_sub$Dice)
mean(HA_PUBMRF_sub$Dice)
mean(HA_JLF_sub$Dice)

mean(HA_majvote_sub_test$Dice)
mean(HA_PUBMRF_sub_test$Dice)
mean(HA_JLF_sub_test$Dice)

t.test(HA_majvote_sub$Dice, HA_PUBMRF_sub$Dice)
t.test(HA_majvote_sub$Dice, HA_JLF_sub$Dice)
t.test(HA_PUBMRF_sub$Dice, HA_JLF_sub$Dice)

df1 <- data.frame(a = c(2, 2,3, 3), b = c(0.800, 0.810, 0.810, 0.800))
df2 <- data.frame(a = c(1, 1,2, 2), b = c(0.785, 0.795, 0.795, 0.785))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.815, label = "**", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 1.5, y = 0.800, label = "*", size = 8)

# HA, CA4 / DG

HA_plot1_CA4 <- HA_plot1[HA_plot1$Label %in% c('4', '104'),]
HA_plot2_CA4 <- HA_plot2[HA_plot2$Label %in% c('4', '104'),]
HA_plot2_CA4$Type <- factor(HA_plot2_CA4$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=HA_plot1_CA4, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.580, 0.800, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(HA_plot2_CA4, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.500, 0.900, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

HA_majvote_CA4 = HA_plot2_CA4[HA_plot2_CA4$Type == "Majority Vote",]
HA_PUBMRF_CA4 = HA_plot2_CA4[HA_plot2_CA4$Type == "PUB-MRF",]
HA_JLF_CA4 = HA_plot2_CA4[HA_plot2_CA4$Type == "JLF",]

HA_majvote_CA4_test <- HA_majvote_CA4[-(1:20), ]
HA_PUBMRF_CA4_test <- HA_PUBMRF_CA4[-(1:20), ]
HA_JLF_CA4_test <- HA_JLF_CA4[-(1:20), ]

mean(HA_majvote_CA4$Dice)
mean(HA_PUBMRF_CA4$Dice)
mean(HA_JLF_CA4$Dice)

mean(HA_majvote_CA4_test$Dice)
mean(HA_PUBMRF_CA4_test$Dice)
mean(HA_JLF_CA4_test$Dice)

t.test(HA_majvote_CA4$Dice, HA_PUBMRF_CA4$Dice)
t.test(HA_majvote_CA4$Dice, HA_JLF_CA4$Dice)
t.test(HA_PUBMRF_CA4$Dice, HA_JLF_CA4$Dice)

df1 <- data.frame(a = c(2, 2,3, 3), b = c(0.842, 0.845, 0.845, 0.842))
df2 <- data.frame(a = c(1, 1,2, 2), b = c(0.837, 0.840, 0.840, 0.837))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.848, label = "*", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 1.5, y = 0.843, label = "*", size = 8)

# HA, CA2 / CA3

HA_plot1_CA23 <- HA_plot1[HA_plot1$Label %in% c('5', '105'),]
HA_plot2_CA23 <- HA_plot2[HA_plot2$Label %in% c('5', '105'),]
HA_plot2_CA23$Type <- factor(HA_plot2_CA23$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=HA_plot1_CA23, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.300, 0.800, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(HA_plot2_CA23, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.300, 0.900, 0.040), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

HA_majvote_CA23 = HA_plot2_CA23[HA_plot2_CA23$Type == "Majority Vote",]
HA_PUBMRF_CA23 = HA_plot2_CA23[HA_plot2_CA23$Type == "PUB-MRF",]
HA_JLF_CA23 = HA_plot2_CA23[HA_plot2_CA23$Type == "JLF",]

HA_majvote_CA23_test <- HA_majvote_CA23[-(1:20), ]
HA_PUBMRF_CA23_test <- HA_PUBMRF_CA23[-(1:20), ]
HA_JLF_CA23_test <- HA_JLF_CA23[-(1:20), ]

mean(HA_majvote_CA23$Dice)
mean(HA_PUBMRF_CA23$Dice)
mean(HA_JLF_CA23$Dice)

mean(HA_majvote_CA23_test$Dice)
mean(HA_PUBMRF_CA23_test$Dice)
mean(HA_JLF_CA23_test$Dice)

t.test(HA_majvote_CA23$Dice, HA_PUBMRF_CA23$Dice)
t.test(HA_majvote_CA23$Dice, HA_JLF_CA23$Dice)
t.test(HA_PUBMRF_CA23$Dice, HA_JLF_CA23$Dice)

df1 <- data.frame(a = c(2, 2,3, 3), b = c(0.722, 0.727, 0.727, 0.722))
df2 <- data.frame(a = c(1, 1,2, 2), b = c(0.712, 0.717, 0.717, 0.712))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.730, label = "*", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 1.5, y = 0.720, label = "*", size = 8)

# HA, sr / sl / sm

HA_plot1_srslsm <- HA_plot1[HA_plot1$Label %in% c('6', '106'),]
HA_plot2_srslsm <- HA_plot2[HA_plot2$Label %in% c('6', '106'),]
HA_plot2_srslsm$Type <- factor(HA_plot2_srslsm$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=HA_plot1_srslsm, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.350, 0.800, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(HA_plot2_srslsm, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.200, 0.900, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

HA_majvote_srslsm = HA_plot2_srslsm[HA_plot2_srslsm$Type == "Majority Vote",]
HA_PUBMRF_srslsm = HA_plot2_srslsm[HA_plot2_srslsm$Type == "PUB-MRF",]
HA_JLF_srslsm = HA_plot2_srslsm[HA_plot2_srslsm$Type == "JLF",]

HA_majvote_srslsm_test <- HA_majvote_srslsm[-(1:20), ]
HA_PUBMRF_srslsm_test <- HA_PUBMRF_srslsm[-(1:20), ]
HA_JLF_srslsm_test <- HA_JLF_srslsm[-(1:20), ]

mean(HA_majvote_srslsm$Dice)
mean(HA_PUBMRF_srslsm$Dice)
mean(HA_JLF_srslsm$Dice)

mean(HA_majvote_srslsm_test$Dice)
mean(HA_PUBMRF_srslsm_test$Dice)
mean(HA_JLF_srslsm_test$Dice)

t.test(HA_majvote_srslsm$Dice, HA_PUBMRF_srslsm$Dice)
t.test(HA_majvote_srslsm$Dice, HA_JLF_srslsm$Dice)
t.test(HA_PUBMRF_srslsm$Dice, HA_JLF_srslsm$Dice)

df1 <- data.frame(a = c(2, 2,3, 3), b = c(0.664, 0.667, 0.667, 0.664))
df2 <- data.frame(a = c(1, 1,2, 2), b = c(0.657, 0.660, 0.660, 0.657))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.670, label = "**", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 1.5, y = 0.663, label = "**", size = 8)

# HA, fimbria

HA_plot1_fi <- HA_plot1[HA_plot1$Label %in% c('11', '37'),]
HA_plot2_fi <- HA_plot2[HA_plot2$Label %in% c('11', '37'),]
HA_plot2_fi$Type <- factor(HA_plot2_fi$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=HA_plot1_fi, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.260, 0.440, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(HA_plot2_fi, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.000, 0.900, 0.050), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

HA_majvote_fi = HA_plot2_fi[HA_plot2_fi$Type == "Majority Vote",]
HA_PUBMRF_fi = HA_plot2_fi[HA_plot2_fi$Type == "PUB-MRF",]
HA_JLF_fi = HA_plot2_fi[HA_plot2_fi$Type == "JLF",]

HA_majvote_fi_test <- HA_majvote_fi[-(1:20), ]
HA_PUBMRF_fi_test <- HA_PUBMRF_fi[-(1:20), ]
HA_JLF_fi_test <- HA_JLF_fi[-(1:20), ]

mean(HA_majvote_fi$Dice)
mean(HA_PUBMRF_fi$Dice)
mean(HA_JLF_fi$Dice)

mean(HA_majvote_fi_test$Dice)
mean(HA_PUBMRF_fi_test$Dice)
mean(HA_JLF_fi_test$Dice)

t.test(HA_majvote_fi$Dice, HA_PUBMRF_fi$Dice)
t.test(HA_majvote_fi$Dice, HA_JLF_fi$Dice)
t.test(HA_PUBMRF_fi$Dice, HA_JLF_fi$Dice)

df1 <- data.frame(a = c(2, 2,3, 3), b = c(0.890, 0.900, 0.900, 0.890))
df2 <- data.frame(a = c(1, 1,2, 2), b = c(0.870, 0.880, 0.880, 0.870))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.905, label = "**", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 1.5, y = 0.885, label = "**", size = 8)

# HA, mammillary body

HA_plot1_mb <- HA_plot1[HA_plot1$Label %in% c('12', '22'),]
HA_plot2_mb <- HA_plot2[HA_plot2$Label %in% c('12', '22'),]
HA_plot2_mb$Type <- factor(HA_plot2_mb$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=HA_plot1_mb, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.740, 0.900, 0.010), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(HA_plot2_mb, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.700, 0.980, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

HA_majvote_mb = HA_plot2_mb[HA_plot2_mb$Type == "Majority Vote",]
HA_PUBMRF_mb = HA_plot2_mb[HA_plot2_mb$Type == "PUB-MRF",]
HA_JLF_mb = HA_plot2_mb[HA_plot2_mb$Type == "JLF",]

HA_majvote_mb_test <- HA_majvote_mb[-(1:20), ]
HA_PUBMRF_mb_test <- HA_PUBMRF_mb[-(1:20), ]
HA_JLF_mb_test <- HA_JLF_mb[-(1:20), ]

mean(HA_majvote_mb$Dice)
mean(HA_PUBMRF_mb$Dice)
mean(HA_JLF_mb$Dice)

mean(HA_majvote_mb_test$Dice)
mean(HA_PUBMRF_mb_test$Dice)
mean(HA_JLF_mb_test$Dice)

t.test(HA_majvote_mb$Dice, HA_PUBMRF_mb$Dice)
t.test(HA_majvote_mb$Dice, HA_JLF_mb$Dice)
t.test(HA_PUBMRF_mb$Dice, HA_JLF_mb$Dice)

df1 <- data.frame(a = c(2, 2,3, 3), b = c(0.982, 0.985, 0.985, 0.982))
df2 <- data.frame(a = c(1, 1,3, 3), b = c(0.967, 0.970, 0.970, 0.967))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.986, label = "**", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 2, y = 0.971, label = "*", size = 8)

# HA, fornix

HA_plot1_fo <- HA_plot1[HA_plot1$Label %in% c('33', '35'),]
HA_plot2_fo <- HA_plot2[HA_plot2$Label %in% c('33', '35'),]
HA_plot2_fo$Type <- factor(HA_plot2_fo$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=HA_plot1_fo, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.360, 0.800, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(HA_plot2_fo, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.300, 0.900, 0.040), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

HA_majvote_fo = HA_plot2_fo[HA_plot2_fo$Type == "Majority Vote",]
HA_PUBMRF_fo = HA_plot2_fo[HA_plot2_fo$Type == "PUB-MRF",]
HA_JLF_fo = HA_plot2_fo[HA_plot2_fo$Type == "JLF",]

HA_majvote_fo_test <- HA_majvote_fo[-(1:20), ]
HA_PUBMRF_fo_test <- HA_PUBMRF_fo[-(1:20), ]
HA_JLF_fo_test <- HA_JLF_fo[-(1:20), ]

mean(HA_majvote_fo$Dice)
mean(HA_PUBMRF_fo$Dice)
mean(HA_JLF_fo$Dice)

mean(HA_majvote_fo_test$Dice)
mean(HA_PUBMRF_fo_test$Dice)
mean(HA_JLF_fo_test$Dice)

t.test(HA_majvote_fo$Dice, HA_PUBMRF_fo$Dice)
t.test(HA_majvote_fo$Dice, HA_JLF_fo$Dice)
t.test(HA_PUBMRF_fo$Dice, HA_JLF_fo$Dice)

df1 <- data.frame(a = c(2, 2,3, 3), b = c(0.842, 0.847, 0.847, 0.842))
df2 <- data.frame(a = c(1, 1,3, 3), b = c(0.860, 0.865, 0.865, 0.860))
df3 <- data.frame(a = c(1, 1,2, 2), b = c(0.833, 0.838, 0.838, 0.833))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 2.5, y = 0.849, label = "**", size = 8) +
  geom_line(data = df2, aes(x = a, y = b)) + annotate("text", x = 2, y = 0.867, label = "**", size = 8) +
  geom_line(data = df3, aes(x = a, y = b)) + annotate("text", x = 1.5, y = 0.840, label = "*", size = 8)

# HA, alveus

HA_plot1_alv <- HA_plot1[HA_plot1$Label %in% c('111', '222'),]
HA_plot2_alv <- HA_plot2[HA_plot2$Label %in% c('111', '222'),]
HA_plot2_alv$Type <- factor(HA_plot2_alv$Type, levels=c("Majority Vote", "JLF", "PUB-MRF"))

dice<- summaryBy(Dice ~ NumAtlas + NumTemplate + Type, data=HA_plot1_alv, FUN=c(mean, se))
ggplot(dice, aes(x=NumTemplate, y=Dice.mean, color = NumAtlas, group = NumAtlas)) + 
  geom_errorbar(aes(ymin=Dice.mean-Dice.se, ymax=Dice.mean+Dice.se), width=.3) +
  labs(x="Number of Templates", y="Mean Dice Score", color="Number of Atlases") +
  geom_smooth(size=.5, se=FALSE) + geom_point() + facet_grid(~Type) +
  scale_y_continuous(breaks=seq(0.360, 0.800, 0.020), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        legend.text = element_text(size=14), legend.title = element_text(size=18),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=0, r=20, b=0, l=0), legend.position="bottom")

pp <- ggplot(HA_plot2_alv, aes(x=Type, y=Dice, fill="red")) + stat_summary(fun.data=MinMeanSEMMax, geom="boxplot") +
  labs(y="Mean Dice Score", x="Algorithm Used") + theme(legend.position="none") +
  scale_y_continuous(breaks=seq(0.300, 0.800, 0.040), labels=scaleFUN) +
  theme(axis.text.x = element_text(size=14), axis.text.y = element_text(size=14),
        axis.title.x = element_text(size=18, angle=0, margin=margin(t=10, r=0, b=5, l=0)),
        axis.title.y = element_text(size=18, angle=90, margin=margin(t=0, r=10, b=0, l=5)),
        strip.text.x = element_text(size=16, margin=margin(t=8, r=0, b=8, l=0)),
        plot.title = element_text(size=18, margin=margin(t=10, r=0, b=15, l=0)),
        plot.margin=margin(t=10, r=20, b=0, l=0), legend.position="none")

HA_majvote_alv = HA_plot2_alv[HA_plot2_alv$Type == "Majority Vote",]
HA_PUBMRF_alv = HA_plot2_alv[HA_plot2_alv$Type == "PUB-MRF",]
HA_JLF_alv = HA_plot2_alv[HA_plot2_alv$Type == "JLF",]

HA_majvote_alv_test <- HA_majvote_alv[-(1:20), ]
HA_PUBMRF_alv_test <- HA_PUBMRF_alv[-(1:20), ]
HA_JLF_alv_test <- HA_JLF_alv[-(1:20), ]

mean(HA_majvote_alv$Dice)
mean(HA_PUBMRF_alv$Dice)
mean(HA_JLF_alv$Dice)

mean(HA_majvote_alv_test$Dice)
mean(HA_PUBMRF_alv_test$Dice)
mean(HA_JLF_alv_test$Dice)

t.test(HA_majvote_alv$Dice, HA_PUBMRF_alv$Dice)
t.test(HA_majvote_alv$Dice, HA_JLF_alv$Dice)
t.test(HA_PUBMRF_alv$Dice, HA_JLF_alv$Dice)

df1 <- data.frame(a = c(1, 1,2, 2), b = c(0.750, 0.755, 0.755, 0.750))

pp + geom_line(data = df1, aes(x = a, y = b)) + annotate("text", x = 1.5, y = 0.757, label = "*", size = 8)
