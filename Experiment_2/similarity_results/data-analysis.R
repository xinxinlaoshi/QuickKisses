both.base <- read.csv('similarity/base-both.csv')[-1]
ditransitive.base <- read.csv('similarity/base-ditransitive.csv')[-1]
transitive.base <- read.csv('similarity/base-transitive.csv')[-1]

summary<- merge(both.base, ditransitive.base, by =c('event','event.category'))
summary <- merge(summary, transitive.base, by =c('event','event.category'))
names(summary) = c('event','event.category','spearman.both','cosine.both','spearman.ditransitive','cosine.ditransitive','spearman.transitive','cosine.transitive')
summary$meaning.shift <- summary$spearman.transitive - summary$spearman.both
  
#### test the main effect of event category ####
library(Matrix)
library(lme4)

summary$event.category <- factor(summary$event.category, levels = c('punctive count', 'durative mass','durative count'))

# use likelihood ratio test to examine the main effect of event category
m.full <- lm(meaning.shift ~ 1+event.category, data = summary)
m.reduced <- lm(meaning.shift ~ 1, data = summary)
library(zoo)
library(lmtest)
lrtest(m.reduced,m.full)

noPC = subset(summary, ! event.category %in% c('punctive count'))
noDC = subset(summary, ! event.category %in% c('durative count'))
noDM = subset(summary, ! event.category %in% c('durative mass'))

### Durative count vs. durative mass ###
noPC.full <- lm(meaning.shift ~ 1+event.category, data = noPC)
noPC.reduced <- lm(meaning.shift ~ 1, data = noPC)
lrtest(noPC.reduced, noPC.full)

### Punctive count vs. durative mass ### 
noDC.full <- lm(meaning.shift ~ 1+event.category, data = noDC)
noDC.reduced <- lm(meaning.shift ~ 1, data = noDC)
lrtest(noDC.reduced, noDC.full)

### Punctive count vs. durative count
noDM.full <- lm(meaning.shift ~ 1+event.category, data = noDM)
noDM.reduced <- lm(meaning.shift ~ 1, data = noDM)
lrtest(noDM.reduced, noDM.full)

