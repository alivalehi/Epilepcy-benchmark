function GetSelectedTextValue(ddlFruits) {

  var selectedText = ddlFruits.options[ddlFruits.selectedIndex].innerHTML;
  var selectedValue = ddlFruits.value;
  // alert("Selected Text: " + selectedText + " Value: " + selectedValue);
  trace2 = trace(models[selectedValue][1].substr(0,20),parseFloat(models[selectedValue][3]),parseFloat(models[selectedValue][4])); 
  data = [trace1, trace2]
  const divtitle = document.getElementById('titleDiv');
  divtitle.innerHTML ="<h2>" +  models[selectedValue][1]+"</h2>";
  const abstractDiv = document.getElementById('abstractDiv');
  abstractDiv.innerHTML ="<h2>Abstract: </h2><p>"+ models[selectedValue][2]+ "</p>";
  
  // $('#two > div > div:nth-child(2) > div.titleDiv').text()
  // $(".abstractDiv").text(models[selectedValue][3])
  Plotly.newPlot('myDiv', data, layout); 
}


var trace = (modelName, a,b) =>
{
  return {
  x: ['Sensitivity', 'Specificity'],
  y: [a, b],
  name: modelName,
  type: 'bar',
     marker: {
      color: 'red'
    }
  }
};



var trace1 = {
  x: ['Sensitivity', 'Specificity'],
  y: [97, 98],
  name: 'The propose Model',
  type: 'bar',
   marker: {
      color: 'green'
    }
};

var trace2 = trace("The propose Model",97,98);  
var data = [trace1, trace2];

var layout = {barmode: 'group'};

Plotly.newPlot('myDiv', data, layout);

let models = 
[
[
"Zabihi, M.; Kiranyaz, S.; Ince, T.; Gabbouj, M.",
"Patient-specific epileptic seizure detection in long-term EEG recording in paediatric patients with intractable seizures.",
"The contemporary diagnosis of epileptic seizures is dominated by non-invasive EEG signal analysis and classification. In this paper, we propose a patient-specific seizure detection technique, which selects the optimal feature subsets and trains a dedicated classifier for each patient in order to maximize the classification performance. Our method exploits time domain, frequency domain, time-frequency domain and non-linear feature sets. Then, by using Conditional Mutual Information Maximization (CMIM) as the feature selection method the optimal feature subset is chosen over which the Support Vector Machine is trained as the classifier. In this study, both train and test sets contain 50% of seizure and non-seizure segments of the EEG signal. From the CHB-MIT Scalp benchmark EEG dataset, we used the EEG data from four subjects with overall 21 hours of recording. Support Vector Machine (SVM) with linear kernel is used as the classifier. The experimental results show a delicate classification performance over the test set: i.e., an average of 90.62% sensitivity and 99.32% specificity are acquired when all channels and recordings are used to form a composite feature vector. In addition, an average of 93.78% sensitivity and a specificity of 99.05% are obtained using CMIM.",
"90.62",
"99.32"
],
[
"Chen, L.L.; Zhang, J.; Zou, J.Z.; Zhao, C.J.; Wang, G.S.",
"A framework on wavelet-based non-linear features and extreme learning machine for epileptic seizure detection.",
"Background Many investigations based on nonlinear methods have been carried out for the research of seizure detection. However, some of these nonlinear measures cannot achieve satisfying performance without considering the basic rhythms of epileptic EEGs.New method To overcome the defects, this paper proposed a framework on wavelet-based nonlinear features and extreme learning machine (ELM) for the seizure detection. Three nonlinear methods, i.e., approximate entropy (ApEn), sample entropy (SampEn) and recurrence quantification analysis (RQA) were computed from orignal EEG signals and corresponding wavelet decomposed sub-bands separately. The wavelet-based energy was measured as the comparative. Then the combination of sub-band features was fed to ELM and SVM classifier respectively. Results The decomposed sub-band signals show significant discrimination between interictal and ictal states and the union of sub-band features helps to achieve better detection. All the three nonlinear methods show higher sensitivity than the wavelet-based energy analysis using the proposed framework. The wavelet-based SampEn-ELM detector reaches the best performance with a sensitivity of 92.6% and a false detection rate (FDR) of 0.078. Compared with SVM, the ELM detector is better in terms of detection accuracy and learning efficiency. Comparison with existing method(s) The decomposition of original signals into sub-bands leads to better identification of seizure events compared with that of the existing nonlinear methods without considering the time–frequency decomposition. Conclusions The proposed framework achieves not only a high detection accuracy but also a very fast learning speed, which makes it feasible for the further development of the automatic seizure detection system.",
"89",
"93"
],
[
"Zabihi, M.; Kiranyaz, S.; Rad, A.B.; Katsaggelos, A.K.; Gabbouj, M.; Ince",
"T. Analysis of High-Dimensional Phase Space via Poincaré Section for Patient-Specific Seizure Detection.",
"In this paper, the performance of the phase space representation in interpreting the underlying dynamics of epileptic seizures is investigated and a novel patient-specific seizure detection approach is proposed based on the dynamics of EEG signals. To accomplish this, the trajectories of seizure and nonseizure segments are reconstructed in a high dimensional space using time-delay embedding method. Afterwards, Principal Component Analysis (PCA) was used in order to reduce the dimension of the reconstructed phase spaces. The geometry of the trajectories in the lower dimensions is then characterized using Poincaré section and seven features were extracted from the obtained intersection sequence. Once the features are formed, they are fed into a two-layer classification scheme, comprising the Linear Discriminant Analysis (LDA) and Naive Bayesian classifiers. The performance of the proposed method is then evaluated over the CHB-MIT benchmark database and the proposed approach achieved 88.27% sensitivity and 93.21% specificity on average with 25% training data. Finally, we perform comparative performance evaluations against the state-of-the-art methods in this domain which demonstrate the superiority of the proposed method.",
"88.26",
"93.21"
],
[
"Fergus, P.; Hussain, A.J.; Hignett, D.; Aljumeily, D.; Abdel-Aziz, K.; Hamdan, H. ",
"A machine learning system for automated whole-brain seizure detection.",
"Epilepsy is a chronic neurological condition that affects approximately 70 million people worldwide. Characterised by sudden bursts of excess electricity in the brain, manifesting as seizures, epilepsy is still not well understood when compared with other neurological disorders. Seizures often happen unexpectedly and attempting to predict them has been a research topic for the last 30 years. Electroencephalograms have been integral to these studies, as the recordings that they produce can capture the brain’s electrical signals. The diagnosis of epilepsy is usually made by a neurologist, but can be difficult to make in the early stages. Supporting para-clinical evidence obtained from magnetic resonance imaging and electroencephalography may enable clinicians to make a diagnosis of epilepsy and instigate treatment earlier. However, electroencephalogram capture and interpretation is time consuming and can be expensive due to the need for trained specialists to perform the interpretation. Automated detection of correlates of seizure activity generalised across different regions of the brain and across multiple subjects may be a solution. This paper explores this idea further and presents a supervised machine learning approach that classifies seizure and non-seizure records using an open dataset containing 342 records (171 seizures and 171 non-seizures). Our approach posits a new method for generalising seizure detection across different subjects without prior knowledge about the focal point of seizures. Our results show an improvement on existing studies with 88% for sensitivity, 88% for specificity and 93% for the area under the curve, with a 12% global error, using the k-NN classifier.",
88,
88
],
[
"Zhang, C.; Altaf, M.A.B.; Yoo, J.",
"Design and Implementation of an On-Chip Patient-Specific Closed-Loop Seizure Onset and Termination Detection System.",
"This paper presents the design of an area- and energy-efficient closed-loop machine learning-based patient-specific seizure onset and termination detection algorithm, and its on-chip hardware implementation. Application- and scenario-based tradeoffs are compared and reviewed for seizure detection and suppression algorithm and system which comprises electroencephalography (EEG) data acquisition, feature extraction, classification, and stimulation. Support vector machine achieves a good tradeoff among power, area, patient specificity, latency, and classification accuracy for long-term monitoring of patients with limited training seizure patterns. Design challenges of EEG data acquisition on a multichannel wearable environment for a patch-type sensor are also discussed in detail. Dual-detector architecture incorporates two area-efficient linear support vector machine classifiers along with a weight-and-average algorithm to target high sensitivity and good specificity at once. On-chip implementation issues for a patient-specific transcranial electrical stimulation are also discussed. The system design is verified using CHB-MIT EEG database [1] with a comprehensive measurement criteria which achieves high sensitivity and specificity of 95.1% and 96.2%, respectively, with a small latency of 1 s. It also achieves seizure onset and termination detection delay of 2.98 and 3.82 s, respectively, with seizure length estimation error of 4.07 s.",
95.1,
96.2
],
[
"Khan, M.R.; Saadeh, W.; Awais, M.; Altaf, B.",
"A Low Complexity Patient-Specific threshold-based Accelerator for the Grand-Mal Seizure Disorder. Biomed.",
"This paper presents a 2-channel electroencephalograph (EEG) based seizure detection accelerator suitable for long-term continuous monitoring of patients suffering from the Grand-mal seizure disorder. The implementation is based on the novel slope based detection (SBD) algorithm to achieve start and end of seizure detection. The proposed SBD algorithm is verified experimentally using a full FPGA implementation with patients' recordings from Physionet Children Hospital Boston (CHB)-MIT EEG database with real-time seizure, information display on the Android phone through a low-power Bluetooth link. The patient-specific detection with specific threshold results in sensitivity, specificity, system latency, and detection latency of 91.2%, 93.6%, 0.5s, and 29.25 s, respectively, using the CHB-MIT EEG database.",
95.01,
97.97
],[
"Zabihi, M.; Kiranyaz, S.; Ince, T.; Gabbouj, M.",
"M. Patient-specific epileptic seizure detection in long-term EEG recording in paediatricpatients with intractable seizures. ",
"The contemporary diagnosis of epileptic seizures is dominated by non-invasive EEG signal analysis and classification. In this paper, we propose a patient-specific seizure detection technique, which selects the optimal feature subsets and trains a dedicated classifier for each patient in order to maximize the classification performance. Our method exploits time domain, frequency domain, time-frequency domain and non-linear feature sets. Then, by using Conditional Mutual Information Maximization (CMIM) as the feature selection method the optimal feature subset is chosen over which the Support Vector Machine is trained as the classifier. In this study, both train and test sets contain 50% of seizure and non-seizure segments of the EEG signal. From the CHB-MIT Scalp benchmark EEG dataset, we used the EEG data from four subjects with overall 21 hours of recording. Support Vector Machine (SVM) with linear kernel is used as the classifier. The experimental results show a delicate classification performance over the test set: i.e., an average of 90.62% sensitivity and 99.32% specificity are acquired when all channels and recordings are used to form a composite feature vector. In addition, an average of 93.78% sensitivity and a specificity of 99.05% are obtained using CMIM.",
90.62,
99.32
]
,[
"Van Esbroeck, A.; Smith, L.; Syed, Z.; Singh, S.; Karam, Z.N.",
"Multi-task seizure detection: Addressing intra-patient variation in seizure morphologies.",
"The accurate and early detection of epileptic seizures in continuous electroencephalographic (EEG) data has a growing role in the management of patients with epilepsy.Early detection allows for therapy to be delivered at the start of seizures and for caregivers to be notified promptly about potentially debilitating events. The challenge to detecting epileptic seizures, however, is that seizure morphologies exhibit considerable inter-patient and intrapatient variability. While recent work has looked at addressing the issue of variations across different patients (inter-patient variability) and described patient-specific methodologies for eizure detection, there are no examples of systems that can simultaneously address the challenges of inter-patient and intra-patient variations in seizure morphology. In our study, we address this complete goal and describe a multi-task learning approach that trains a classifier to perform well across many kinds of seizures rather than potentially overfitting to the most common seizure types. Our approach increases the generalizability of seizure detection systems and improves the tradeoff between latency and sensitivity versus false positive rates. When compared against the standard approach on the CHB–MIT multi-channel scalp EEG data, our proposed method improved discrimination between seizure and non-seizure EEG for almost 83% of the patients while reducing false positives on nearly 70% of the patientsstudied.",
100,
93
],
[
"Fergus, P.; Hignett, D.; Hussain, A.; Al-jumeily, D.; Abdel-aziz, K. ",
"Automatic Epileptic Seizure Detection Using Scalp EEG and Advanced Artificial Intelligence Techniques. ",
"The epilepsies are a heterogeneous group of neurological disorders and syndromes characterised by recurrent, involuntary, paroxysmal seizure activity, which is often associated with a clinicoelectrical correlate on the electroencephalogram. The diagnosis of epilepsy is usually made by a neurologist but can be difficult to be made in the early stages. Supporting paraclinical evidence obtained from magnetic resonance imaging and electroencephalography may enable clinicians to make a diagnosis of epilepsy and investigate treatment earlier. However, electroencephalogram capture and interpretation are time consuming and can be expensive due to the need for trained specialists to perform the interpretation. Automated detection of correlates of seizure activity may be a solution. In this paper, we present a supervised machine learning approach that classifies seizure and nonseizure records using an open dataset containing 342 records. Our results show an improvement on existing studies by as much as 10% in most cases with a sensitivity of 93%, specificity of 94%, and area under the curve of 98% with a 6% global error using a k-class nearest neighbour classifier. We propose that such an approach could have clinical applications in the investigation of patients with suspected seizure disorders.",
94,
97.27
],
[
"Khan, Y.U.; Rafiuddin, N.; Farooq, O.",
"Automated seizure detection in scalp EEG using multiple wavelet scales. In Proceedings of the 2012 IEEE International Conference on Signal Processing, Computing and Control",
"The proposed research work designs a detector algorithm for automatic detection of epileptic seizures. In this work a wavelet based feature extraction technique has been adopted. Epochs of EEG are decomposed using discrete wavelet transform (DWT) up to 5 level of wavelet decomposition. Relative values of energy and a normalized coefficient of variation (NCOV) based measure, (σ 2 /μ a ) are computed on the wavelet coefficients acquired in the frequency range of 0-32 Hz from both seizure and non-seizure segments. The performance of NCOV over the traditionally used coefficient of variation, COV (σ 2 /μ 2 ) was studied. The feature NCOV yielded better performance than the commonly used COV, σ 2 /μ 2 . The algorithm was evaluated on 5 subjects from CHB-MIT scalp EEG database.",
83.6,
100
],
[
"Samiee, K.; Kiranyaz, S.; Gabbouj, M.; Saramäki, T.",
"Long-term epileptic EEG classification via 2D mapping and textural features",
"Interpretation of long-term Electroencephalography (EEG) records is a tiresome task for clinicians. This paper presents an efficient, low cost and novel approach for patient-specific classification of long-term epileptic EEG records. We aim to achieve this with the minimum supervision from the neurologist. To accomplish this objective, first a novel feature extraction method is proposed based on the mapping of EEG signals into two dimensional space, resulting into a texture image. The texture image is constructed by mapping and scaling EEG signals and their associated frequency sub-bands into the gray-level image domain. Image texture analysis using gray level co-occurrence matrix (GLCM) is then applied in order to extract multivariate features which are able to differentiate between seizure and seizure-free events. To evaluate the discriminative power of the proposed feature extraction method, a comparative study is performed, against other dedicated feature extraction methods. The comparative performance evaluations show that the proposed feature extraction method can outperform other state-of-art feature extraction methods with a low computational cost. With a training rate of , the overall sensitivity of and specificity of are achieved in the classification of over 163 hours of EEG records using support vector machine (SVM) classifiers with linear kernels and trained by the stochastic gradient descent (SGD) algorithm.",
70.19,
97.74
],
[
"Zhang, A.; Altaf, M.A.B.; Yoo, J.",
"A 16-channel, 1-Second Latency Patient-Specific Seizure Onset and Termination Detection Processor with Dual Detector Architecture and Digital Hysteresis. In Proceedings of the Custom Integrated Circuit Conference",
"This paper presents an area-power-efficient 16-channel seizure onset and termination detection processor with patient-specific machine learning techniques. This is the first work in literature to report an on-chip classification to detect both start and end of seizure event simultaneously with high accuracy. Frequency-Time Division Multiplexing (FTDM) filter architecture and Dual-Detector Architecture (D2A) is proposed, implemented and verified. The D2A incorporates two area-efficient Linear Support Vector Machine (LSVM) classifiers along with digital hysteresis to achieve a high sensitivity and specificity of 95.7% and 98%, respectively, using CHB-MIT EEG database [1], with a small latency of 1s. The overall energy efficiency is measured as 1.85μJ/Classification at 16-channel mode.",
95.7,
98
],
[
"Orosco, L.; Correa, A.G.; Diez, P.; Laciar, E.",
"Patient non-specific algorithm for seizures detection in scalp EEG.",
"Epilepsy is a brain disorder that affects about 1% of the population in the world. Seizure detection is an important component in both the diagnosis of epilepsy and seizure control. In this work a patient non-specific strategy for seizure detection based on Stationary Wavelet Transform of EEG signals is developed. A new set of features is proposed based on an average process. The seizure detection consisted in finding the EEG segments with seizures and their onset and offset points. The proposed offline method was tested in scalp EEG records of 24–48 h of duration of 18 epileptic patients. The method reached mean values of specificity of 99.9%, sensitivity of 87.5% and a false positive rate per hour of 0.9.",
87.5,
99.5
],
[
"Bugeja, S.; Garg, L.; Audu, E.E.",
"A novel method of EEG data acquisition, feature extraction and feature space creation for Early Detection of Epileptic Seizures",
"In this paper we describe a simple and very fast method of data acquisition, feature extraction and feature space creation for epileptic seizure detection. The scalp electroencephalogram (EEG) dataset [1, 2] collected at the Children's Hospital Boston from 22 pediatric patients having 192 intractable seizures (available as CHB-MIT database) is used to assess this simple approach against existing ones [1, 3], with very positive results reaching up to 99.48% Sensitivity.",
97.98,
89.90
],
[
"Janjarasjitt, S.",
"Epileptic seizure classifications of single-channel scalp EEG data using wavelet-based features and SVM. Med. Biol.",
"In this study, wavelet-based features of single-channel scalp EEGs recorded from subjects with intractable seizure are examined for epileptic seizure classification. The wavelet-based features extracted from scalp EEGs are simply based on detail and approximation coefficients obtained from the discrete wavelet transform. Support vector machine (SVM), one of the most commonly used classifiers, is applied to classify vectors of wavelet-based features of scalp EEGs into either seizure or non-seizure class. In patient-based epileptic seizure classification, a training data set used to train SVM classifiers is composed of wavelet-based features of scalp EEGs corresponding to the first epileptic seizure event. Overall, the excellent performance on patient-dependent epileptic seizure classification is obtained with the average accuracy, sensitivity, and specificity of, respectively, 0.9687, 0.7299, and 0.9813. The vector composed of two wavelet-based features of scalp EEGs provide the best performance on patient-dependent epileptic seizure classification in most cases, i.e., 19 cases out of 24. The wavelet-based features corresponding to the 32-64, 8-16, and 4-8 Hz subbands of scalp EEGs are the mostly used features providing the best performance on patient-dependent classification. Furthermore, the performance on both patient-dependent and patient-independent epileptic seizure classifications are also validated using tenfold cross-validation. From the patient-independent epileptic seizure classification validated using tenfold cross-validation, it is shown that the best classification performance is achieved using the wavelet-based features corresponding to the 64-128 and 4-8 Hz subbands of scalp EEGs.",
72.99,
98.13
],
[
"Bhattacharyya, A.; Pachori, R.B.",
"A Multivariate Approach for Patient-Specific EEG Seizure Detection Using Empirical Wavelet Transform. IEEE Trans.",
"Objective: This paper investigates the multivariate oscillatory nature of electroencephalogram (EEG) signals in adaptive frequency scales for epileptic seizure detection. Methods: The empirical wavelet transform (EWT) has been explored for the multivariate signals in order to determine the joint instantaneous amplitudes and frequencies in signal adaptive frequency scales. The proposed multivariate extension of EWT has been studied on multivariate multicomponent synthetic signal, as well as on multivariate EEG signals of Children's Hospital Boston-Massachusetts Institute of Technology (CHB-MIT) scalp EEG database. In a moving-window-based analysis, 2-s-duration multivariate EEG signal epochs containing five automatically selected channels have been decomposed and three features have been extracted from each 1-s part of the 2-s-duration joint instantaneous amplitudes of multivariate EEG signals. The extracted features from each oscillatory level have been processed using a proposed feature processing step and joint features have been computed in order to achieve better discrimination of seizure and seizure-free EEG signal epochs. Results: The proposed detection method has been evaluated over 177 h of EEG records using six classifiers. We have achieved average sensitivity, specificity, and accuracy values as 97.91%, 99.57%, and 99.41%, respectively, using tenfold cross-validation method, which are higher than the compared state of art methods studied on this database. Conclusion: Efficient detection of epileptic seizure is achieved when seizure events appear for long duration in hours long EEG recordings. Significance: The proposed method develops time-frequency plane for multivariate signals and builds patient-specific models for EEG seizure detection.",
97.91,
99.57
],
[
"Janjarasjitt, S.",
"Performance of epileptic single-channel scalp EEG classifications using single wavelet-based features.",
"Classification of epileptic scalp EEGs are certainly ones of the most crucial tasks in diagnosis of epilepsy. Rather than using multiple quantitative features, a single quantitative feature of single-channel scalp EEG is applied for classifying its corresponding state of the brain, i.e., during seizure activity or non-seizure period. The quantitative features proposed are wavelet-based features obtained from the logarithm of variance of detail and approximation coefficients of single-channel scalp EEG signals. The performance on patient-dependent based epileptic seizure classifications using single wavelet-based features are examined on scalp EEG data of 12 children subjects containing 79 seizures. The 4-fold cross validation is applied to evaluate the performance on patient-dependent based epileptic seizure classifications using single wavelet-based features. From the computational results, it is shown that the wavelet-based features can provide an outstanding performance on patient-dependent based epileptic seizure classification. The average accuracy, sensitivity, and specificity of patient-dependent based epileptic seizure classification are, respectively, 93.24%, 83.34%, and 93.53%.",
83.34,
93.53
],
[
"Alickovic, F.; Kevric, J.; Subasi, A.",
"Performance evaluation of empirical mode decomposition, discrete wavelet transform, and wavelet packed decomposition for automated epileptic seizure detection and prediction.",
"This study proposes a new model which is fully specified for automated seizure onset detection and seizure onset prediction based on electroencephalography (EEG) measurements. We processed two archetypal EEG databases, Freiburg (intracranial EEG) and CHB-MIT (scalp EEG), to find if our model could outperform the state-of-the art models. Four key components define our model: (1) multiscale principal component analysis for EEG de-noising, (2) EEG signal decomposition using either empirical mode decomposition, discrete wavelet transform or wavelet packet decomposition, (3) statistical measures to extract relevant features, (4) machine learning algorithms. Our model achieved overall accuracy of 100% in ictal vs. inter-ictal EEG for both databases. In seizure onset prediction, it could discriminate between inter-ictal, pre-ictal, and ictal EEG with the accuracy of 99.77%, and between inter-ictal and pre-ictal EEG states with the accuracy of 99.70%. The proposed model is general and should prove applicable to other classification tasks including detection and prediction regarding bio-signals such as EMG and ECG.",
99.65,
99.8
],
[
"Tsiouris, K.; Pezoulas, V.C.; Zervakis, M.; Konitsiotis, S.; Koutsouris, D.D.; Fotiadis, D.I.",
"A Long Short-Term Memory deep learning network for the prediction of epileptic seizures using EEG signals",
"The electroencephalogram (EEG) is the most prominent means to study epilepsy and capture changes in electrical brain activity that could declare an imminent seizure. In this work, Long Short-Term Memory (LSTM) networks are introduced in epileptic seizure prediction using EEG signals, expanding the use of deep learning algorithms with convolutional neural networks (CNN). A pre-analysis is initially performed to find the optimal architecture of the LSTM network by testing several modules and layers of memory units. Based on these results, a two-layer LSTM network is selected to evaluate seizure prediction performance using four different lengths of preictal windows, ranging from 15 min to 2 h. The LSTM model exploits a wide range of features extracted prior to classification, including time and frequency domain features, between EEG channels cross-correlation and graph theoretic features. The evaluation is performed using long-term EEG recordings from the open CHB-MIT Scalp EEG database, suggest that the proposed methodology is able to predict all 185 seizures, providing high rates of seizure prediction sensitivity and low false prediction rates (FPR) of 0.11–0.02 false alarms per hour, depending on the duration of the preictal window. The proposed LSTM-based methodology delivers a significant increase in seizure prediction performance compared to both traditional machine learning techniques and convolutional neural networks that have been previously evaluated in the literature.",
99.84,
99.86
],[
"Deng, Z.; Xu, P.; Xie, L.; Choi, K.S.; Wang, S.",
"Transductive Joint-Knowledge-Transfer TSK FS for Recognition of Epileptic EEGSignals.",
"Intelligent recognition of electroencephalogram (EEG) signals is an important means to detect seizure. Traditional methods for recognizing epileptic EEG signals are usually based on two assumptions: 1) adequate training examples are available for model training and 2) the training set and the test set are sampled from data sets with the same distribution. Since seizures occur sporadically, training examples of seizures could be limited. Besides, the training and test sets are usually not sampled from the same distribution for generic non-patient-specific recognition of EEG signals. Hence, the two assumptions in traditional recognition methods could hardly be satisfied in practice, which results in degradation of model performance. Transfer learning is a feasible approach to tackle this issue attributed to its ability to effectively learn the knowledge from the related scenes (source domains) for model training in the current scene (target domain). Among the existing transfer learning methods for epileptic EEG recognition, transductive transfer learning fuzzy systems (TTL-FSs) exhibit distinctive advantages-the interpretability that is important for medical diagnosis and the transfer learning ability that is absent from traditional fuzzy systems. Nevertheless, the transfer learning ability of TTL-FSs is restricted to a certain extent since only the discrepancy in marginal distribution between the training data and test data is considered. In this paper, the enhanced transductive transfer learning Takagi-Sugeno-Kang fuzzy system construction method is proposed to overcome the challenge by introducing two novel transfer learning mechanisms: 1) joint knowledge is adopted to reduce the discrepancy between the two domains and 2) an iterative transfer learning procedure is introduced to enhance transfer learning ability. Extensive experiments have been carried out to evaluate the effectiveness of the proposed method in recognizing epileptic EEG signals on the Bonn and CHB-MIT EEG data sets. The results show that the method is superior to or at least competitive with some of the existing state-of-art methods under the scenario of transfer learning.",
91.91,
93.16
],
[
"Sopic, D.; Aminifar, A.; Atienza, D",
"A Wearable System for Real-Time Detection of Epileptic Seizures in Children. I",
"Today, epilepsy is one of the most common chronic diseases affecting more than 65 million people worldwide and is ranked number four after migraine, Alzheimer's disease, and stroke. Despite the recent advances in anti-epileptic drugs, one-third of the epileptic patients continue to have seizures. More importantly, epilepsy-related causes of death account for 40% of mortality in high-risk patients. However, no reliable wearable device currently exists for real-time epileptic seizure detection. In this paper, we propose e-Glass, a wearable system based on four electroencephalogram (EEG) electrodes for the detection of epileptic seizures. Based on an early warning from e-Glass, it is possible to notify caregivers for rescue to avoid epilepsy-related death due to the underlying neurological disorders, sudden unexpected death in epilepsy, or accidents during seizures. We demonstrate the performance of our system using the Physionet.org CHB-MIT Scalp EEG database for epileptic children. Our experimental evaluation demonstrates that our system reaches a sensitivity of 93.80% and a specificity of 93.37%, allowing for 2.71 days of operation on a single battery charge.",
93.60,
93.37
]
]