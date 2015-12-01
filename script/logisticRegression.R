# Logistic Regression model
library(tm)
library(stringr)
library(slam)
library(glmnet)

setwd("C:/UVA/DataMining/SYS6018_customerservice/labels/3_final_labels")

# Get the tweet file
tweets <- read.csv("trainData.csv", stringsAsFactors = T)

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

# Get rid of useless tweet
tweets[tweets$tokens == "",] # None of these are complaints. It is safe to get rid of them
tweets <- tweets[tweets$tokens != "",]

# Corpus
docvar <- list(industry="industry", content="tokens", complaint="complaint")
myReader <- readTabular(mapping=docvar)
corpus <- Corpus(DataframeSource(tweets), readerControl=list(reader=myReader))

# Clean up corpus
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation) 
corpus <- tm_map(corpus, removeWords, c(stopwords("english"), "https",'southwestair', "delta", "united", "deltaassist", "americanair", "jetblue", "comcast", "comcastcares","verizonsupport","vzwsupport", "verizon", "att", "attcares", "tmobilehelp", "dish", "hulu_support", "dish_answers", "hulu", "tmobile"))
corpus <- tm_map(corpus, removeNumbers) 
corpus <- tm_map(corpus, stripWhitespace)

# Document term matrix
dtm = DocumentTermMatrix(corpus, control = list(minWordLength = 3, bounds=list(global=c(5,5125))))
dtm
corpus <- corpus[row_sums(dtm) > 0] # Note that this makes us lose 7 complaints out of 120 tweets
dtm <- dtm[row_sums(dtm) > 0,] # And we suppress the documents that are now empty
dtm

# Convert to matrix form and add the label "complaint"
dtm.m <- cbind(unlist(meta(corpus, "complaint")), as.matrix(dtm))

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
log(bestlam) # what's in the plot

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
testPred <- predict(trainll, newx = test[,-1], s = bestlam, type="class")
table(testPred, test[,1])

# Compute metrics of performance

# Accuracy
sum(testPred == test[,1]) / length(testPred) # 83% Accuracy

# Recall
sum(testPred == 1 & testPred == test[,1]) / sum(test[,1] == 1) # 55% recall

# Precision
sum(testPred == 1 & testPred == test[,1]) / sum(testPred == 1) # 77% precision