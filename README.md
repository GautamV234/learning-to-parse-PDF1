#  ES 654 - Learning to parse PDF

_Gautam Vashishtha_   
_Krishnam Hasija_  
_Praveen Venkatesh_  
_Rwik Rana_  
_Shreyshi Singh_   

The repo contains all the relevant code for the testing the results in the final report.

The PDF format has established itself as the \textit{de-facto} standard for transferring documents. The PDF format is popularized in the contemporary era as it makes sharing of lengthy documents quite simple via a single globally accessible file. However, PDFs restructure data to pursue universality, making it harder to convert them back to an editable state. PDFs can be created using various tools, making the conversion to an editable format complex. Editable PDFs would make the searching, data extraction and indexing of documents much more manageable. By developing a parser capable of extracting data from semi-structured sources such as PDFs, we may acquire relatively easy access to the abundance of material accessible on the World Wide Web.

# Layouts
Layouts contains the code for the segmentation models which are used for inference namely CTPN and UNET.   
To run inference on CTPN run: python ctpn_predict.py [with the appropriate args]  
To run inference on UNET run: testunet.ipynb  

# OCR
OCR contains the code which are used for Text recognition  
1)CNN_S2S - CNN Encoder RNN Decoder Seq to Seq model  
To run inference - python test.py [with the appropriate args]  
2)ViT_math - Vision Transformer for Math recongition  
To run inference - python test.py [with the appropriate args]  
3)ViT_ocr - Vision Transformer for Text recognition  
To run inference - python test.py [with the appropriate args]  

# Reading Order
Reading Order contains the code for determining the reading order of the parsed document  
1)Ngram - Using beam search with n=2 on the generated Ngrams to find the reading order  
To run the code - run the notebook ngram.ipynb  
2)Content Vectors using BERT - Generating reading order using context vector similarity between paragraphs using bert  
To run the code - run the notebook sbert.ipynb  

# Website to test Math to Latex Model!  
You can run the interactive website and test different images containing LaTex and get the corresponding latex output from it.  
To run the code-  
cd streamlit_website/latex_ocr  
python web.py and open the localhost port to see the website  

The report for the project can be found here -> https://drive.google.com/file/d/1qPDmtJxfDg8_bDIRpEzOLu4r7_Qps_1D/view?usp=sharing
