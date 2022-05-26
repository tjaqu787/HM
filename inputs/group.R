library(data.table)
data<-fread("~/Documents/projects/HM/inputs/art_target.csv",sep=",",nrows = 1)

cx_groupby <- function(){
  data<-fread("~/Documents/projects/HM/inputs/preferences.csv",sep=",")
  x <- names(data)[14:ncol(data)]
  y <- names(data)[1]

  print('grouping')
  flush.console()
  df<- data[, lapply(.SD,sum), .SDcols=x, by=y]
  data = df[df$customer_id,]
  print('normalizing')
  flush.console()
  df <- df[,(.SD / rowSums(.SD)), .SDcols=x,by=y]
  df[df$customer_id,]=data
  print('writing')
  flush.console()
  fwrite(df,file = '~/Documents/projects/HM/inputs/pref.csv')
  print('done')
  flush.console()
}


art_groupby <- function(){
  data<-fread("~/Documents/Projects/Kaggle comps/HM_recommendation/inputs/clothing.csv",sep=",")


  x <- names(data)[14:ncol(data)]
  y <- names(data)[1]

  df<- data[, lapply(.SD,sum), .SDcols=x, by=y]
  
  fwrite(df,file = '~/Documents/Projects/Kaggle comps/HM_recommendation/inputs/art_pref.csv')
}

cx_groupby()
