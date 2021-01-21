""" Test """

## Image upload 후 실행
test_image=Image.open('IMG_4260.jpg')
test_image=test_image.resize((224,224))
test_image=np.array(test_image)
test_image=test_image/255.

## original image
plt.imshow(test_image)
plt.show()

test_image=np.reshape(test_image,(1,224,224,3))

prediction1=new_model.predict(test_image)

pred1=np.zeros_like(prediction1)
thr=0.5
pred1[prediction1>=thr]=1
pred1[prediction1<thr]=0

## segmentation 후 image
plt.figure(figsize=(11,5))
plt.subplot(121)
plt.imshow(test_image[0])
plt.subplot(122)
plt.imshow(pred1[0,:,:,1])
plt.show()
