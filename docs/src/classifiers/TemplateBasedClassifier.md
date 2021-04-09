# TemplateBasedClassifier  
  
- - -    
## mseTopN\_templates( histograms, templates, n=-1 )  
**calculate the mse between each histogram in histograms and each histogram in templates**  
input arguments:  
- histograms: 2D numpy array of shape (nhistograms, nbins)  
- templates: 2D numpy array of shape (ntemplates,nbins)  
- n: integer representing the number of (sorted) bin squared errors to take into account (default: all)  
output:  
2D numpy array of shape (nhistograms,ntemplates) holding the mseTopN between each  
  
- - -    
## mseTopN\_min( histograms, templates, n=-1 )  
**calculate the mse betwee a histogram and each template and return the minimum**  
input arguments:  
- histograms: 2D numpy array of shape (nhistograms, nbins)  
- templates: 2D numpy array of shape (ntemplates,nbins)  
- n: integer representing the number of (sorted) bin squared errors to take into account (default: all)  
output:  
1D numpy array of shape (nhistograms) holding the minimum mseTopN for each histogram  
  
- - -    
## mseTop10\_min( histograms, templates )  
**special case of above with n=10**  
  
- - -    
## mseTopN\_avg( histograms, templates, n=-1 )  
**calculate the mse betwee a histogram and each template and return the average**  
input arguments:  
- histograms: 2D numpy array of shape (nhistograms, nbins)  
- templates: 2D numpy array of shape (ntemplates,nbins)  
- n: integer representing the number of (sorted) bin squared errors to take into account (default: all)  
output:  
1D numpy array of shape (nhistograms) holding the average mseTopN for each histogram  
  
- - -    
## mseTop10\_avg( histograms, templates )  
**special case of above with n=10**  
  
- - -    
## TemplateBasedClassifier(HistogramClassifier)  
**histogram classifier based on a direct comparison with templates (i.e. reference histograms)**  
  
### \_\_init\_\_( self, templates, comparemethod='minmse' )  
**initializer from a set of templates (reference histograms)**  
input arguments:  
- templates: a 2D numpy array of shape (nhistograms,nbins)  
- comparemethod: string representing the method by which to compare a histogram with a set of templates  
currently supported methods are:  
- minmse: minimum mean square error between histogram and all templates  
- avgmse: average mean square error between histogram and all templates  
  
### evaluate( self, histograms )  
**classification of a collection of histograms based on their deviation from templates**  
  
