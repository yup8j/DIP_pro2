# Digital Image Process Project 2 Task 2

Classify the pore images.

## Installation

```
git clone https://github.com/yup8j/DIP_pro2.git

cd DIP_pro2

pip install -r requirements.txt
```
Note: [OpenCV](https://docs.opencv.org/3.4.5/) is required. 
## Run model

```
python pro2.py --testing [your test image folder]
```

### Example

```
python3 pro2.py --testing testimg                               
```

### Output

```
testimg/32.jpg pores
testimg/104.jpg pores
testimg/405.jpg background
testimg/103.jpg pores
testimg/401.jpg background
testimg/102.jpg pores
testimg/406.jpg background
testimg/402.jpg background
testimg/404.jpg background
testimg/407.jpg background
testimg/106.jpg pores
testimg/403.jpg background
testimg/107.jpg pores
```

### Test image

32.jpg ![32.jpg](testimg/32.jpg)

104.jpg ![104.jpg](testimg/104.jpg)

405.jpg ![405.jpg](testimg/405.jpg)

103.jpg ![103.jpg](testimg/103.jpg)
