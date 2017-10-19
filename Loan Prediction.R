library('caret')
library('nnet')
library('e1071')
library('plyr')
library('gtools')

#Load the datasets
train <- read.csv('Train.csv',header = TRUE, sep = ",",na.strings = c(""," "))
test <- read.csv('Test.csv',header = TRUE, sep = ",",na.strings = c(""," "))

Loan_ID <- test$Loan_ID
result <- data.frame(Loan_ID)
Loan_Status <- train$Loan_Status

train_modified <- train[,-which(names(train) %in% c('Loan_Status'))]
data <- rbind (train_modified,test)
data <- data[,-which(names(data) %in% c('Loan_ID'))]


#Convert to factor
names <- c('Loan_Amount_Term','Credit_History')
data[,names] <- lapply(data[,names] , factor)


for (i in names(data)){
  if (is.factor(data[[i]])){
    vector <- data[[i]]
    if (any(is.na(vector))){
      levels(vector) <- c(levels(vector),'Unknown')
      vector[is.na(vector)] <- 'Unknown'
      data[[i]] <- vector
    }
  }
}



dummy <- dummyVars(~., data = data, fullRank = T)
data <- data.frame(predict(dummy,data))



col_names <- sapply(data, function(col) length(unique(col)) < 10)
data[ , col_names] <- lapply(data[ , col_names] , factor)



train_modified <- data[1:nrow(train),]
test_modified <-data[(nrow(train)+1):nrow(data),]



#Replace missing values and Normalize continuous variables
train_norm <- predict(preProcess(train_modified,method= c("medianImpute","range")),train_modified)
test_norm <- predict(preProcess(test_modified,method = c("medianImpute","range")),test_modified)

train_norm <- cbind(train_norm, Loan_Status)

train_norm <- train_norm[,-which(names(train_norm) %in% c('Married.Unknown', 
              'Loan_Amount_Term.12', 'Loan_Amount_Term.12','Loan_Amount_Term.36',
              'Loan_Amount_Term.60','Loan_Amount_Term.84','Loan_Amount_Term.120',
              'Loan_Amount_Term.240','Loan_Amount_Term.300','Loan_Amount_Term.350',
              'Loan_Amount_Term.480','Loan_Amount_Term.Unknown'))]

test_norm <- test_norm[,-which(names(test_norm) %in% c('Married.Unknown', 
              'Loan_Amount_Term.12', 'Loan_Amount_Term.12','Loan_Amount_Term.36',
              'Loan_Amount_Term.60','Loan_Amount_Term.84','Loan_Amount_Term.120',
              'Loan_Amount_Term.240','Loan_Amount_Term.300','Loan_Amount_Term.350',
              'Loan_Amount_Term.480','Loan_Amount_Term.Unknown'))]
                                 

              

#Data Partition of train into train and validation
rows <- sample(nrow(train_norm),floor(nrow(train_norm)*0.70))
train_norm_train <- train_norm[rows,]
train_norm_valid <- train_norm[-rows,]



#Enlist Models
models <- c('SVM','Logistic Regression')

modelForm <- function (train_norm_train,model_name){

  if (model_name == 'Logistic Regression')
    model <- glm(Loan_Status ~., train_norm_train, family = 'binomial')

  else if (model_name == 'SVM')
    model <- svm(Loan_Status ~ ., train_norm_train, type = 'nu-classification',
                 kernel = 'radial')

  return(model)

}

prediction <- function(model,train_norm_valid,model_name){

  if (model_name == 'Logistic Regression'){
    predictedValues <- predict(model,train_norm_valid, type = 'response')
    predictedValues <- ifelse(predictedValues > 0.5,'Y','N')
  }

  else{
    predictedValues <- predict(model,train_norm_valid)
  }

  return(predictedValues)
}

#Create Models, make predictions on validation dataset and display confusion matrix
for (model_name in models){
  model <- modelForm(train_norm_train,model_name)
  predicted <- prediction(model,train_norm_valid,model_name)
  cm <- confusionMatrix(predicted,train_norm_valid$Loan_Status)
  print(paste("Accuracy for", model_name))
  print (as.matrix(cm$overall['Accuracy']))

  predicted_test<- prediction(model,test_norm,model_name)
  result <- cbind(result,'Loan_Status'= predicted_test)
}

write.csv(result[,c(1:2)], file = 'Submission_SVM.csv',row.names = FALSE)
write.csv(result[,c(1,3)], file = 'Submission_Logistic_Regression.csv',row.names = FALSE)
