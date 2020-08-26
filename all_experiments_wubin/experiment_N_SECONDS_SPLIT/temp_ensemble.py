     ...:         thres = np.percentile(y,90,axis=0)      
     ...:         above_mean = y> thres                   
     ...:         ans = y * above_mean                    
     ...:         #ye = np.mean(y, axis=0)                
     ...:         yel = []                                
     ...:         for i in range(y.shape[1]):             
     ...:             total_weight = (ans[:,i] !=0).sum() 
     ...:             total = sum(ans[:,i])               
     ...:             avg = total/total_weight            
     ...:             yel.append(avg)                     
     ...:         ye = np.array(yel)                      
     ...:                                                 
     ...:                                                 
     ...:         pred = cfg.LABELS[np.argmax(ye)]        
     ...:                                              
