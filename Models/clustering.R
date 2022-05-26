library(data.table)
library(genieclust)

genders =c('inputs/male_cx_data.csv','inputs/female_cx_data.csv','inputs/family_cx_info.csv')
df = fread(genders[3])
df[is.na(df)] <- 0
full = gclust(df)



df = fread('customer_info.csv')
head(fa)
head(m)
head(w)
fa$fam_order = fam$order

df = merge(df,f[,c('customer_id','full_order')],by = 'customer_id',all=TRUE)
df = merge(df,m[,c('customer_id','mens_order')],by = 'customer_id',all=TRUE)
df = merge(df,w[,c('customer_id','womens_order')],by = 'customer_id',all=TRUE)
df = merge(df,fa[,c('customer_id','fam_order')],by = 'customer_id',all=TRUE)
df[is.na(df)] <- 0

head(df)

fwrite(df,'customer_info.csv')
