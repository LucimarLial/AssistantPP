#Any possible outlier obtained by this method should be examined in the context of the purpose of the dataset.

#Now we can identify which points are outside this range, that is, they can be considered outliers.

markdown_outliers = '''
The method of removing outilers used in this project is the Amplitude Interquartile: 

    1 - Calculates the interquartile interval for the data;
    2 - Multiply the interquartile interval (IQR) by the number 1.5;
    3 - Add 1.5 x (IQR) for the third quartile. Any larger number is a possible outlier; and
    4 - Subtract 1.5 x (IQR) from the first quartile. Any number smaller than this is a possible outlier.
    
**Everything outside the range [Q1 - 1.5 X IQR, Q3 + 1.5 X IQR] is considered an anomalous point for that pattern.**

'''

markdown_missing_values = '''
Machine Learning algorithms are not able to handle missing values (missing data). 
In most cases, assigning a reasonable estimate of a suitable data value is better than leaving it blank.
Automatic approaches that can be used:

i) create a new value for the qualitative attribute that indicates that the value was unknown; e 

ii) use statistical measures for quantitative attributes, such as: average, mode or median of known values.
'''

markdown_class_desbalance = '''

In classification problems, when there is a marked variation in the number of objects between the classes in the target column the data set is considered unbalanced, for example: classes A and B have the proportion of 80:20 or 90:10. 

This situation may cause the model to bias, i.e., the model to be adjusted too much for the samples of the majority class.

In practice the model will respond very well to the majority class samples, but will perform poorly for the minority class samples.

'''

markdown_class_desbalance_v2 = '''
Sampling is a preprocessing that aims to minimize the discrepancies between the sample quantities of the classes of the data set, by means of a re-sampling. with the purpose of generating a balanced data set. Techniques used to redefine the size of the data set:

* Oversampling: creates new samples of the minority class from the information contained in the original data. This generation of new samples can be done randomly with the help of clustering techniques or synthetically.
**Undersampling**: reduces the unbalancing of the data set by randomly eliminating samples from the majority class. 
'''

markdown_class_desbalance_v3 = '''
**Oversampling** replicates the already existing data, increasing the number of instances of minority classes. **The advantage is that no information is discarded**, but the **computing cost will be high**.

**Undersampling** extracts a random subset of the majority class, **preserving the characteristics of the class**, being ideal for situations of large volumes of data. Although it reduces computational and storage time, **this technique discards majority-class information**, which can lead to lower performance in its predictions.
'''

markdown_binning = '''
**Discretization**

Operation that transforms quantitative data (contínuous) into qualitative data, that is, numeric attributes into discrete or nominal attributes with a finite number of intervals, obtaining an unimposed partition of a continuous domain. An association between each interval with a discrete numeric value is then established. Once the discretization is performed, the data can be treated as nominal data.

Bins (buckets or intervals) are created that contain approximately the same amount of observations - quantile strategy.
'''
# or that are equally spaced - uniform strategy.

markdown_scaling = '''
**Normalization**

It consists of adjusting the scale of the values of each attribute so that the values are in small ranges, such as -1 to 1 or 0 to 1. 

It is recommended when the lower and upper limits of attribute values are very different, which leads to a large variation of values, or even when several attributes are on different scales, to avoid that one attribute prevails over another. 

**Linear Standardization - MinMaxScaler**

To put in the $[0, 1]$ range, just subtract each value from the minimum value and divide by the difference of the maximum and minimum value:

**Xscaled = x - min(x) / max(x) - min(x)**

'''

markdown_standardization = '''
**Standardization by Standard Deviation - Standardization **

Standardization by standardization is best for dealing with outiliers and standardizes the scale of the data without interfering with its shape. It is useful for classifiers, especially those who work with distance.

It consists of making the variable zero mean and variance one, to subtract the mean from the data for each observation and divide by the standard deviation:

** Xstandardized = x - x (sample mean) / s **

where ** s ** is the sample standard deviation. 
'''

markdown_onehot = '''
**OneHot Encoder**

The encoding aims to transform the value domains of certain attributes of the data set.

One of the simplest forms of representation of categorical variables is through the method called OneHot Enconding. With it, a categorical variable with $h$ categories is transformed into $h$ new binary variables (0 or 1), where the presence of 1 (hot) means that that observation belongs to that category and 0 (cold) that does not.
'''

markdown_ordinal = '''
**Ordinal Encoder**

In this method the values are converted to ordinal integers. This results in a single column of integers (0 to n_categories - 1) per attribute.
'''
