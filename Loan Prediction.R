library('caret')
library('nnet')
library('e1071')
library('plyr')
library('randomForest')
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

train_modified <- data[1:nrow(train),]
train_modified <- predict(preProcess(train_modified,
                    method = c('medianImpute')),train_modified)

#boruta_model <- Boruta(Loan_Status ~., train_modified)
#final.model <- TentativeRoughFix(boruta_model)
#variables <- getSelectedAttributes(final.model, withTentative = F)
#train_modified <- cbind(train_modified, Loan_Status)


test_modified <-data[(nrow(train)+1):nrow(data),]


dummy_train <- dummyVars(~., data = train_modified, fullRank = T)
train_dummy <- data.frame(predict(dummy_train,train_modified))

dummy_test <- dummyVars(~., data = test_modified, fullRank = T)
test_dummy <- data.frame(predict(dummy_test,test_modified))


col_names <- sapply(train_dummy, function(col) length(unique(col)) < 10)
train_dummy[ , col_names] <- lapply(train_dummy[ , col_names] , factor)

col_names <- sapply(test_dummy, function(col) length(unique(col)) < 10)
test_dummy[ , col_names] <- lapply(test_dummy[ , col_names] , factor)



#Replace missing values and Normalize continuous variables
train_norm <- predict(preProcess(train_dummy,method= c("medianImpute","range")),train_dummy)
test_norm <- predict(preProcess(test_dummy,method = c("medianImpute","range")),test_dummy)


#Based on Boruta Feature Selection
variables_added <- c('ApplicantIncome','CoapplicantIncome','LoanAmount','Credit_History.1', 'Credit_History.Unknown',
                     'Property_Area.Semiurban','Property_Area.Urban')

train_norm <- train_norm[,which (names(train_norm) %in% c(variables_added))]
train_norm <- cbind(train_norm, Loan_Status)

test_norm <- test_norm[, which (names(test_norm) %in% c(variables_added))]


#Data Partition of train into train and validation
rows <- sample(nrow(train_norm),floor(nrow(train_norm)*0.80))
train_norm_train <- train_norm[rows,]
train_norm_valid <- train_norm[-rows,]



#Enlist Models
models <- c('SVM','Logistic Regression', 'Random Forest')

modelForm <- function (train_norm_train,model_name){

  if (model_name == 'Logistic Regression')
    model <- train(Loan_Status ~., train_norm, method = 'glm', 
                   family = 'binomial',
                   trControl = trainControl(
                     method = "cv", number = 10,
                     verboseIter = FALSE
                   ))
  
  else if (model_name == 'SVM')
    model <- train(Loan_Status ~., train_norm, method = 'svmRadial',
                   trControl = trainControl(
                     method = "cv", number = 10,
                     verboseIter = FALSE
                   ))
  
  else if (model_name == 'Random Forest')
    model <- train(Loan_Status ~., train_norm, method = 'rf',
                   trControl = trainControl(
                     method = "cv", number = 10,
                     verboseIter = FALSE
                   ))
  
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
  print(paste("Accuracy for", model_name))
  print (confusionMatrix(model,'average'))

  #predicted_test<- prediction(model,test_norm,model_name)
  #result <- cbind(result,'Loan_Status'= predicted_test)
}

#write.csv(result[,c(1:2)], file = 'Submission_SVM.csv',row.names = FALSE)
#write.csv(result[,c(1,3)], file = 'Submission_Logistic_Regression.csv',row.names = FALSE)
#write.csv(result[,c(1,4)], file = 'Submission_Random_Forest.csv',row.names = FALSE)
