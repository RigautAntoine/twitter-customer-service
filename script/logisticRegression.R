# Logistic Regression model
library(tm)
library(stringr)
library(slam)
library(glmnet)
library(ROCR)
library(ggplot2)
library(caret)
library(RColorBrewer)
library(AUC)


CL=brewer.pal(9, "Set1")

# supporting functions
plot.ROC.curve <- function(probs, labels){
  preds <- prediction(probs, labels)
  perf <- performance(preds, measure = "tpr", x.measure = "fpr")
  auc <- performance(preds, measure = "auc")
  auc <- auc@y.values[[1]]
  
  roc.data <- data.frame(fpr=unlist(perf@x.values),
                         tpr=unlist(perf@y.values),
                         model="GLM")
  ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
    geom_ribbon(alpha=0.2) +
    geom_line(aes(y=tpr)) +
    ggtitle(paste0("ROC Curve w/ AUC=", auc))
}

ROC <- function(probs, labels){
  preds <- prediction(probs, labels)
  perf <- performance(preds, measure = "tpr", x.measure = "fpr")
  auc <- performance(preds, measure = "auc")
  auc <- auc@y.values[[1]]
  return(list(roc=cbind(unlist(perf@x.values),unlist(perf@y.values)),
              auc=auc))
}


plot.CV.ROC.curve = function(formula, data, k){
  folds <- createFolds(1:nrow(data), k)
  plot(0:1,0:1,col="white",xlab="",ylab="")
  auc <- rep(0, k)
  for(i in 1:k){
    train <- data[unlist(folds[-i]),]
    test <- data[unlist(folds[i]),]
    fit <- glm(formula, data=train, family = "binomial")
    fit.probs <- predict(fit, newdata = test, type = "response")
    cur <- ROC(fit.probs, test[,"complaint"])
    lines(cur$roc,type="s",col=CL[i])
    auc[i] <- cur$auc
  }
  title(main = paste0("Mean AUC: ", mean(auc)))
}



# Set the directory
setwd("C:/UVA/DataMining/SYS6018_customerservice/labels/3_final_labels")

# Get the tweet file
tweets <- read.csv("trainData.csv", stringsAsFactors = T)

# Get the sentiment scores
scores <- read.csv("sentimentScores.csv", stringsAsFactors = F)
names(scores)[5] <- "sentiment"
scores <- scores[,c("text", "industry", "sentiment", "complaint")]

# Naive model
set.seed(1) 
ind <- sample(nrow(scores), 3000)
naive.train <- scores[ind,]
naive.test <- scores[-ind,]
naive <- glm(complaint~sentiment, data=naive.train, family = "binomial")
summary(naive)
naive.probs <- predict(naive, newdata = naive.test, type = "response")

# Plot an AUC curve
plot.ROC.curve(naive.probs, naive.test$complaint)

# Plot cross-validated AUC curves
plot.CV.ROC.curve(complaint~sentiment, scores, 5)

# Compute metrics of performance
naive.preds <- sapply(naive.probs, function(x){ifelse(x > 0.5, 1, 0)})
table(naive.preds, naive.test$complaint) 
# Our naive model is very conservative about classifying tweets as complaints.
# We have a lot of false negatives. 

# Accuracy
sum(naive.preds == naive.test$complaint) / length(naive.preds) # 74% Accuracy
# Recall
sum(naive.preds == 1 & naive.preds == naive.test$complaint) / sum(naive.test$complaint == 1) # 0.5% recall
# Precision
sum(naive.preds == 1 & naive.preds == naive.test$complaint) / sum(naive.preds == 1) # 30% precision


##############################################################

# Logistic Regression and Bag-of-Words representation of Tweets

##############################################################

# Get tweets as parsed by the POS tagger
POStags <- read.csv("POStags.csv", sep = "\t", header = F, stringsAsFactors = F)
names(POStags) <- c("token", "tag", "confidence", "original")

# Preprocess each tweet
tweets$tokens <- apply(POStags, 1, function(x){
  tags <- unlist(str_split(x[2]," "))
  tokens <- unlist(str_split(x[1]," "))
  confidence <- as.numeric(gsub(",", ".", unlist(str_split(x[3], " "))))
  return(paste(tokens[(tags %in% c("N", "^", "Z", "V", "M", "A", "R", "Y", "#")) & confidence > 0.6], collapse=" "))
})

# Add the sentiment score
tweets$sentiment <- scores$sentiment

# Get rid of useless tweet
tweets[tweets$tokens == "",] # None of these are complaints. It is safe to get rid of them
tweets <- tweets[tweets$tokens != "",]

# Corpus
docvar <- list(industry="industry", content="tokens", complaint="complaint", sentiment="sentiment")
myReader <- readTabular(mapping=docvar)
corpus <- Corpus(DataframeSource(tweets), readerControl=list(reader=myReader))

# Clean up corpus
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation) 
corpus <- tm_map(corpus, removeWords, c(stopwords("english"), "https",'southwestair', "comcastdoesntcare", "americanairlines", "delta", "united", "deltaassist", "americanair", "jetblue", "comcast", "comcastcares","verizonsupport","vzwsupport", "verizon", "att", "attcares", "tmobilehelp", "dish", "hulu_support", "dish_answers", "hulu", "tmobile", "comcastsucks"))
corpus <- tm_map(corpus, removeNumbers) 
corpus <- tm_map(corpus, stemDocument) 
corpus <- tm_map(corpus, stripWhitespace)

# Document term matrix
dtm = DocumentTermMatrix(corpus, control = list(minWordLength = 3, bounds=list(global=c(5,5125))))
dtm
corpus <- corpus[row_sums(dtm) > 0] # Note that this makes us lose 7 complaints out of 120 tweets
dtm <- dtm[row_sums(dtm) > 0,] # And we suppress the documents that are now empty
dtm

# Convert to matrix form and add the label "complaint"
dtm.m <- cbind(unlist(meta(corpus, "complaint")), unlist(meta(corpus, "sentiment")), as.matrix(dtm))
df <- data.frame( complaint = unlist(meta(corpus, "complaint")), sentiment = unlist(meta(corpus, "sentiment")), as.matrix(dtm))

# Construct training and testing sets
set.seed(1)
ind <- sample(nrow(dtm.m), 3000) 
train <- dtm.m[ind,] # Train has about 70% of the set
test <- dtm.m[-ind,] # Test has the remaining 30%

# Logistic regression with regularization
cv.lam <- cv.glmnet(train[,-1], factor(train[,1]), alpha=1, family="binomial", type.measure = "class")
plot(cv.lam)
bestlam <- cv.lam$lambda.min # best lambda as selected by cross validation
bestlam
log(bestlam)

# Estimate lasso logistic with lambda chosen by cv on training data
trainll <- glmnet(train[,-1], factor(train[,1]), alpha=1, family="binomial")

# Get the sorted non-zero coefficients
trainll.coef <- predict(trainll, type="coefficients", s=bestlam)
coef.df <- data.frame(word = unlist(trainll.coef@Dimnames[1]), as.matrix(trainll.coef))
coef.df$word <- as.character(coef.df$word)
coef.df <- coef.df[coef.df$X1 != 0,]
coef.df <- coef.df[order(coef.df$X1, decreasing = T),]
View(coef.df)

# How does it work on test set?
probs <- predict(trainll, newx = test[,-1], s = bestlam, type="response")
testPred <- predict(trainll, newx = test[,-1], s = bestlam, type="class")
table(testPred, test[,1])

# Compute metrics of performance

# Accuracy
sum(testPred == test[,1]) / length(testPred) # 83% Accuracy

# Recall
sum(testPred == 1 & testPred == test[,1]) / sum(test[,1] == 1) # 55% recall

# Precision
sum(testPred == 1 & testPred == test[,1]) / sum(testPred == 1) # 77% precision

# AUC curve
plot.ROC.curve(probs, test[,1])

# Separate into industries

dtm.m <- data.frame(industry=unlist(meta(corpus, "industry")), dtm.m)

air.dtm <- as.matrix(dtm.m[dtm.m$industry == "airlines", -1])
tel.dtm <- as.matrix(dtm.m[dtm.m$industry == "telecom", -1])

# Construct training and testing sets
set.seed(1)
ind <- sample(nrow(air.dtm), 1500) 
air.train <- air.dtm[ind,] # Train has about 70% of the set
air.test <- air.dtm[-ind,] # Test has the remaining 30%

# Logistic regression with regularization
air.cv.lam <- cv.glmnet(air.train[,-1], factor(air.train[,1]), alpha=1, family="binomial", type.measure = "class")
air.bestlam <- air.cv.lam$lambda.min # best lambda as selected by cross validation

# Estimate lasso logistic with lambda chosen by cv on training data
air.trainll <- glmnet(air.train[,-1], factor(air.train[,1]), alpha=1, family="binomial")
air.probs <- predict(air.trainll, newx = air.test[,-1], s = air.bestlam, type="response")


sum(testPred == 1 & testPred == test[,1]) / sum(test[,1] == 1) # 55% recall

# AUC curve
air.preds <- prediction(air.probs, air.test[,1])
air.perf <- performance(air.preds, measure = "tpr", x.measure = "fpr")
air.auc <- performance(air.preds, measure = "auc")
air.auc <- air.auc@y.values[[1]]

air.roc.data <- data.frame(fpr=unlist(air.perf@x.values),
                       tpr=unlist(air.perf@y.values),
                       model="GLM")
ggplot(air.roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", air.auc))


# Telecom industry
# Construct training and testing sets
set.seed(1)
ind <- sample(nrow(tel.dtm), 1500) 
tel.train <- tel.dtm[ind,] # Train has about 70% of the set
tel.test <- tel.dtm[-ind,] # Test has the remaining 30%

# Logistic regression with regularization
tel.cv.lam <- cv.glmnet(tel.train[,-1], factor(tel.train[,1]), alpha=1, family="binomial", type.measure = "class")
tel.bestlam <- tel.cv.lam$lambda.min # best lambda as selected by cross validation

# Estimate lasso logistic with lambda chosen by cv on training data
tel.trainll <- glmnet(tel.train[,-1], factor(tel.train[,1]), alpha=1, family="binomial")
tel.probs <- predict(tel.trainll, newx = tel.test[,-1], s = tel.bestlam, type="response")

# AUC curve
tel.preds <- prediction(tel.probs, tel.test[,1])
tel.perf <- performance(tel.preds, measure = "tpr", x.measure = "fpr")
tel.auc <- performance(tel.preds, measure = "auc")
tel.auc <- tel.auc@y.values[[1]]

tel.roc.data <- data.frame(fpr=unlist(tel.perf@x.values),
                           tpr=unlist(tel.perf@y.values),
                           model="GLM")
ggplot(tel.roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", tel.auc))


plot(perf, col="red", main = "ROC curve")
plot(air.perf, add = TRUE, col="green")
plot(tel.perf, add = TRUE, col="blue")
