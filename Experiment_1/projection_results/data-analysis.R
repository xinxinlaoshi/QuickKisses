base.d = read.csv('condition-i/ditransitive.csv')[-1]
base.t = read.csv('condition-i/transitive.csv')[-1]
# base.d = read.csv('condition-ii/ditransitive.csv')[-1]
# base.t = read.csv('condition-ii/transitive.csv')[-1]

proj.base <- merge(base.d, base.t, by = c('event','event.category'))
names(proj.base) <- c('event','event.category', 'ditransitive','transitive')
data <- melt(proj.base, id = c('event','event.category','model'))
colnames(data)[4] <- 'construction'
colnames(data)[5] <- 'scores'

# build a mix-effects model
m.full <- lmer(scores~ construction + event.category+ (1|event), data = data)
m.reduced <- lmer(scores ~  event.category +(1|event), data = data)
m.reduced1 <- lmer(scores ~ construction + (1|event), data = data)
m.interact <- lmer(scores~ construction*event.category + (1|event), data = data)

# see if there is a main effect for construction
anova(m.full, m.reduced)

# see if there is a main effect for event category
anova(m.full, m.reduced1)

# see if there is a main effect for interaction term
anova(m.full, m.interact)

# pairwise comparison between the transitive and ditransitive frames 
PC = subset(all.base, event.category %in% c('punctive count'))
DC = subset(all.base, event.category %in% c('durative count'))
DM = subset(all.base, event.category %in% c('durative mass'))

# see if there is a main effect for construction in the group of the punctive count events
PC_m <- lmer(scores ~ 1+ construction +(1|event), data=PC)
PC_m0 <- lmer(scores ~ 1 +(1|event) , data=PC)
anova(PC_m0,PC_m)

# see if there is a main effect for construction in the group of the durative count events
DC_m <- lmer(scores ~ 1+ construction+(1|event), data=DC)
DC_m0 <- lmer(scores ~ 1+(1|event), data=DC)
anova(DC_m0,DC_m)

# see if there is a main effect for construction in the group of the durative mass events
DM_m <- lmer(scores ~ 1+ construction +(1|event), data=DM)
DM_m0 <- lmer(scores ~ 1+(1|event), data=DM)
anova(DM_m0,DM_m)



