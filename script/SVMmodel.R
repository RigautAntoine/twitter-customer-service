# Data Mining Final Project
# SMV Classification Model
# Dylan Greenleaf (djg3cg)

load("SYS6018_customerservice/labels/3_final_labels/trainingData.RData")
library(e1071)
library(tm)
library(topicmodels)
library(stringr)
library(slam)
library(ROCR)
library(ggplot2)

# Remove words that are in less that 10 tweets
df_label_sent <- df[,c(1,2)]
df_words <- df[,c(-1,-2)]
word_count <- apply(df_words,2, sum)
count <- length(word_count[word_count > 10]) # Removing words that are in 10 or fewer tweets leaves 625 features
df_words_reduced <- df_words[, which(word_count > 10)]

# Now create new dataframe with reduced number of words
df_reduced <- cbind(df_label_sent, df_words_reduced)

# Construct training and testing sets
set.seed(1)
ind <- sample(nrow(df_reduced), 3500) 
train <- df_reduced[ind,] # Train has about 70% of the set
test <- df_reduced[-ind,] # Test has the remaining 30%


# Now ready to try fitting SVM model
# Determine best tuning parameters
# perform a grid search
tuneResult <- tune(svm, complaint ~., data = train, ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9)))
print(tuneResult)
# best performance: MSE = 8.371412, RMSE = 2.89 epsilon 1e-04 cost 4
# Draw the tuning graph
plot(tuneResult)

# There seem to be features that are throwing errors when fitting the model
# Find those words(features) and remove.
findOffendingCharacter <- function(x, maxStringLength=256){
  i <<- i+1
  print(x)
  print(i)
  for (c in 1:maxStringLength){
    offendingChar <- substr(x,c,c)
    #print(offendingChar) #uncomment if you want the indiv characters printed
    #the next character is the offending multibyte Character
  }    
}

string_vector <- colnames(df_reduced)
i <-  0
lapply(string_vector, findOffendingCharacter)
# Remove offending feature name
df_reduced <- df_reduced[, -i]



### NEW APPROACH ###
# Use topic modeling to reduce dimensionality

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

# Get the sentiment scores
scores <- read.csv("sentimentScores.csv", stringsAsFactors = F)
names(scores)[5] <- "sentiment"
scores <- scores[,c("text", "industry", "sentiment", "complaint")]

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
corpus <- tm_map(corpus, removeWords, c(stopwords("english"), "https",'southwestair', "americanairlines", "delta", "united", "deltaassist", "americanair", "jetblue", "comcast", "comcastcares","verizonsupport","vzwsupport", "verizon", "att", "attcares", "tmobilehelp", "dish", "hulu_support", "dish_answers", "hulu", "tmobile", "comcastsucks"))
corpus <- tm_map(corpus, removeNumbers) 
corpus <- tm_map(corpus, stripWhitespace)

# Document term matrix
dtm = DocumentTermMatrix(corpus, control = list(minWordLength = 3, bounds=list(global=c(5,5125))))
dtm
corpus <- corpus[row_sums(dtm) > 0] # Note that this makes us lose 7 complaints out of 120 tweets
dtm <- dtm[row_sums(dtm) > 0,] # And we suppress the documents that are now empty
dtm

# Run LDA
tweet.topic.model = LDA(dtm, 25)
tm.gamma <- tweet.topic.model@gamma
tm.terms <- terms(tweet.topic.model, 20)

new_df <- cbind(df[,c('complaint','sentiment')], tm.gamma)

# Construct training and testing sets
set.seed(1)
ind <- sample(nrow(new_df), 3500) 
train <- new_df[ind,] # Train has about 70% of the set
test <- new_df[-ind,] # Test has the remaining 30%


# Now ready to try fitting SVM model
# Determine best tuning parameters
# perform a grid search
tuneResult <- tune(svm, complaint ~., data = train, ranges = list(epsilon = seq(0,1,0.1), cost = 2^(2:9)))
print(tuneResult)
# best performance: MSE = 0.155, RMSE = 0.3931365 epsilon 0.4 cost 4
# Draw the tuning graph
plot(tuneResult)

# Rerun the tuning optimization given the previous result
# Determine best tuning parameters
# perform a grid search
tuneResult_2 <- tune(svm, complaint ~., data = train, ranges = list(epsilon = seq(0.35,0.45,0.01), cost = 2^(2:4)))
print(tuneResult_2)
# best performance: MSE = 0.154, RMSE = 0.3913192 epsilon 0.44 cost 4
# Draw the tuning graph
png("svm_tune.png", width = 600, height = 600)
plot(tuneResult_2)
dev.off()

svm_model <- tuneResult$best.model
summary(svm_model)

# How does it work on test set?
probs <- predict(svm_model, newdata = test[,-1], type="response")
testPred <- ifelse(probs >= 0.25, 1, 0)
table(testPred, test[,1])

# Compute metrics of performance

# Accuracy
sum(testPred == test[,1]) / length(testPred) # 71% Accuracy

# Recall
sum(testPred == 1 & testPred == test[,1]) / sum(test[,1] == 1) # 73% recall

# Precision
sum(testPred == 1 & testPred == test[,1]) / sum(testPred == 1) # 50% precision

# AUC curve
preds <- prediction(probs, test[,1])
perf <- performance(preds, measure = "tpr", x.measure = "fpr")
auc <- performance(preds, measure = "auc")
auc <- auc@y.values[[1]]

roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="GLM")

png("SMV_ROC.png", width = 600, height = 600)
ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))
dev.off()

save.image(file = "smv_model.RData")
