from sklearn.model_selection import train_test_split    
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, random_state=66, shuffle=True,
    # x, y, shuffle=False,
    train_size=0.8
)   
x_val, x_test, y_val, y_test = train_test_split(    
    x_test, y_test, random_state=66,
    # x_test, y_test, shuffle=False,
    test_size=0.4
)        
