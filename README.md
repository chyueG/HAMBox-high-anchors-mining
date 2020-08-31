# hambox
hambox: Delving into Online High-quality Anchors Mining for Detecting Outer Faces
done:
1 online high quality achor mining 
  a little different from the paper,assume there is an outer face ,compansented anchor and anchor matched in the fisrt step should be top K in candidates,
  the sort algorithm may be difficult，so after calculte iou outerface and decoded regression,erase M anchors which match in the first step,the sort the left 
  anchors and select k- m as compensated anchors
2 regression focal loss
3 data anchor sample

todo
1 paper refers use p2-p6 six layers,anchor size is 0.68*[16,32,64,128,256,512], re-implement anchor size dont take ratio 0.68 into account
2 train loss is convergence , but during training ,dead lock always appears,the training code is under going
