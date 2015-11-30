# Process the final labels to a single training set

setwd("C:/UVA/DataMining/SYS6018_customerservice/labels/3_final_labels")

df1 <- read.csv("airlines_set1_finalLabels.csv")
df2 <- read.csv("airlines_set2_finalLabels.csv")
df3 <- read.csv("telecom_set1_finalLabels.csv")
df4 <- read.csv("telecom_set2_finalLabels.csv")

data <- data.frame(rbind(df1, df2, df3, df4))
data$industry <- as.factor(c(rep("airlines", 3000), rep("telecom", 3000)))
data$text <- as.character(data$text)
names(data)[3] <- "complaint"

copy = data

tweets <- data.frame(unique(data$text))
names(tweets) <- "text"

tweets$complaint <- apply(tweets, 1, function(tweet){
  mean(data[data$text == tweet["text"],"complaint"])
})

tweets[72, "complaint"] <- 1
tweets[262, "complaint"] <- 0
tweets[270, "complaint"] <- 1
tweets[369, "complaint"] <- 1
tweets[482, "complaint"] <- 1
tweets[704, "complaint"] <- 0
tweets[712, "complaint"] <- 1
tweets[1248, "complaint"] <- 0
tweets[1383, "complaint"] <- 1
tweets[2798, "complaint"] <- 1
tweets[2833, "complaint"] <- 0

tweets$industry <- apply(tweets, 1, function(tweet){
  dat <- data[data$text == tweet["text"],"industry"]
  return(dat[1])
})

write.csv(tweets, "trainData.csv", row.names = F)
