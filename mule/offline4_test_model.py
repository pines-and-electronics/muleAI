
current_model = this_model


#%%
preds = list()
for i,img in enumerate(X_train):
    img2 = np.expand_dims(img,0)
    pred = current_model.predict(img2)
    #print(i, pred)
    preds.append(pred[0][0])

#%%
    
#np.array(preds)

pred_matrix = pred_arr.reshape(len(pred_arr), 1)


res= pd.DataFrame([y,pred_matrix])

#np.con

pred_series = pd.Series(pred_matrix.flatten())
y_series = pd.Series(y.flatten())
res = pd.DataFrame({'y':y_series, 'pred':pred_series})