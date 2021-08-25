fs = require('fs');

// Models
let rf = 'Random Forest'
let et = 'Extra Trees'
let lme = 'Linear Mixed Effect'
let linreg = 'Linear Regression'
let svm = 'Support Vector Machines'
let idw = 'Inverse Distance Weighting'
let krig = 'Krigging'
let down = 'Downscaling'
let gam = 'Generalized Additive Model'
let nonlinear = 'Non-linear exposure lag response'
let ctm = 'Chemical Transport Model'
let tfem = 'Time-fixed-effects model'
let ensemble = 'Ensemble'
let gb = 'Gradient Boost'
let mars = 'Multivariate adaptive regression splines'
let xgboost = 'XGBoost'
let cubist = 'Cubist'
let lasso = 'Least Absolute Shrinkage and Selection Operator'
let tsam = 'Timely structure adaptive modeling'
let adaboost = 'Ada Boost'

let bagging = 'Bagging'
let decision_tree = 'Decision Tree'
let knn = 'K-Nearest Neighbors'

// Neural Networks
let mlp = 'Multi-Layer Perceptron'
let cnn = 'Convolutional Neural Network'
let deep = 'Deep learning'
let deepbelief = 'Deep beleif neural network'
let lstm = 'Long-short-term memory neural network'
let brnn= 'Bayesian Regularized Neuranl Network'
let rbfnet = 'Radial Basis Function Network'

// Data characteristics
let dropped = 'Dropped'
let interp = 'Interpolated'

let modis = 'MODIS (Moderate Resolution Imaging Spectroradiometer)'
let maic = 'MAIAC (Multi-Angle Implementation of Atmospheric Correction)'
let dbdt = 'Deep Blue / Dark Target'
let collect5 = 'Collection 5'

let hima = 'Himawari 8'

let point = 'point'
let one_km = '1km'
let three_km = '3km'
let five_km = '5km'
let ten_km = '10km'
let one_hundred_m = '100m'
let twelve_km = '12km'
let twentyseven_km = '27km'
let point_5_by_point_sixtwofive = '0.5° × 0.625°'
let point_5_by_point_5 = '0.5° × 0.5°'
let zero_poin_zero_zero_five = '0.05°'

let one_day = '1 day'
let one_hour = '1 hour'
let multi_hour = 'Multi-hour'
let one_month = '1 month'

let papers = [
    {
        'title': 'Estimating PM2.5 concentrations in Northeastern China with full spatiotemporal coverage, 2005–2016',
        'author': '(Meng et al. 2021)',
        'year': 2021,
        'models': [rf],
        'data': {
            'spatial resolution': [one_km],
            'temporal resolution': [one_day],
            'missing': [interp],
            'training': {
                'start': 'January 2013',
                'end': 'December 2016',
            },
            'testing': {
                'start': 'January 2005',
                'end': 'December 2012',
            },
            'AOD': {
                'source': modis,
                'algorithm': maic
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://www.sciencedirect.com/science/article/pii/S0034425720305769'
    },
    {
        'title': 'A comparison of statistical and machine learning methods for creating national daily maps of ambient PM2.5 concentration',
        'author': '(Berrocal et al. 2020)',
        'year': 2020,
        'models': [linreg, idw, rf, svm, mlp, down],
        'data': {
            'spatial resolution': [twelve_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2011',
                'end': 'December 2011',
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': null,
                'algorithm': null
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://www.sciencedirect.com/science/article/pii/S1352231019307691'
    },
    {
        'title': 'A machine learning method to estimate PM2.5 concentrations across China with remote sensing, meteorological and land use information',
        'author': '(Chen et al. 2018)',
        'year': 2018,
        'models': [rf, gam, nonlinear],
        'data': {
            'spatial resolution': [ten_km],
            'temporal resolution': [one_day],
            'missing': [interp],
            'training': {
                'start': 'January 2012',
                'end': 'December 2016'
            },
            'testing': {
                'start': 'January 2005',
                'end': 'December 2016'
            },
            'AOD': {
                'source': modis,
                'algorithm': dbdt
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'http://www.sciencedirect.com/science/article/pii/S0048969718314281'
    },
    {
        'title': 'A model framework to reduce bias in ground-level PM2.5 concentrations inferred from satellite-retrieved AOD',
        'author': '(Yao and Palmer 2021)',
        'year': 2021,
        'models': [linreg, rf, tfem],
        'data': {
            'spatial resolution': [twentyseven_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2014',
                'end': 'December 2014'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': maic
            },
            'land use': false,
            'meteorological': true
        },
        'url' : 'https://www.sciencedirect.com/science/article/pii/S1352231021000352'
    },
    {
        'title': 'A novel calibration approach of MODIS AOD data to predict PM<sub>2.5</sub> concentrations',
        'author': '(Lee et al. 2011)',
        'year': 2011,
        'models': [lme],
        'data': {
            'spatial resolution': [ten_km],
            'temporal resolution': [one_day],
            'missing': [lme],
            'training': {
                'start': 'January 2003',
                'end': 'December 2003'
            },
            'testing': {
                'start': 'January 2003',
                'end': 'December 2003'
            },
            'AOD': {
                'source': modis,
                'algorithm': collect5
            },
            'land use': false,
            'meteorological': false
        },
        'url' : 'https://acp.copernicus.org/articles/11/7991/2011/'
    },
    {
        'title': 'A Spatial-Temporal Interpretable Deep Learning Model for improving interpretability and predictive accuracy of satellite-based PM2.5',
        'author': '(Yan et al. 2021)',
        'year': 2021,
        'models': [deep],
        'data': {
            'spatial resolution': [three_km, ten_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2016',
                'end': 'December 2018',
            },
            'testing': {
                'start': 'January 2019',
                'end': 'December 2019',
            },
            'AOD': {
                'source': modis,
                'algorithm': dbdt
            },
            'land use': false,
            'meteorological': true
        },
        'url' : 'https://www.sciencedirect.com/science/article/pii/S0269749121000373'
    },
    {
        'title': 'Ambient Air Pollution Exposure Estimation for the Global Burden of Disease 2013',
        'author': '(Brauer et al. 2016)',
        'year': 2016,
        'models': [],
        'data': {
            'spatial resolution': [ten_km],
            'temporal resolution': [one_hour],
            'missing': [],
            'training': {
                'start': null,
                'end': null
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': null,
                'algorithm': null
            },
            'land use': null,
            'meteorological': null
        },
        'url' : 'https://doi.org/10.1021/acs.est.5b03709'
    },
    {
        'title': 'An Ensemble Machine-Learning Model To Predict Historical PM2.5 Concentrations in China from Satellite Data',
        'author': '(Xiao et al. 2018)',
        'year': 2018,
        'models': [rf, gam, xgboost],
        'data': {
            'spatial resolution': [point_5_by_point_sixtwofive],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2013',
                'end': 'December 2016'
            },
            'testing': {
                'start': 'January 2017',
                'end': 'July 2017',
            },
            'AOD': {
                'source': modis,
                'algorithm': dbdt
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://doi.org/10.1021/acs.est.8b02917'
    },
    {
        'title': 'Assessing PM2.5 concentrations in Tehran, Iran, from space using MAIAC, deep blue, and dark target AOD and machine learning algorithms',
        'author': '(Nabavi et al. 2019)',
        'year': 2019,
        'models': [gb, mars, rf, svm],
        'data': {
            'spatial resolution': [ten_km],
            'temporal resolution': [one_hour],
            'missing': [],
            'training': {
                'start': 'January 2011',
                'end': 'December 2016'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': maic
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'http://www.sciencedirect.com/science/article/pii/S1309104218306032'
    },
    {
        'title': 'Assessing temporally and spatially resolved PM2.5 exposures for epidemiological studies using satellite aerosol optical depth measurements',
        'author': '(Kloog et al. 2011)',
        'year': 2011,
        'models': [lme],
        'data': {
            'spatial resolution': [ten_km],
            'temporal resolution': [one_day],
            'missing': [dropped],
            'training': {
                'start': 'January 2000',
                'end': 'December 2008'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': collect5
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://www.sciencedirect.com/science/article/pii/S1352231011009125'
    },
    {
        'title': 'Development of Europe-Wide Models for Particle Elemental Composition Using Supervised Linear Regression and Random Forest',
        'author': '(Chen et al. 2020)',
        'year': 2020,
        'models': [linreg, rf],
        'data': {
            'spatial resolution': [one_hundred_m],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'October 2008',
                'end': 'April 2011'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': null,
                'algorithm': null
            },
            'land use': true,
            'meteorological': false
        },
        'url' : 'https://doi.org/10.1021/acs.est.0c06595'
    },
    {
        'title': 'Estimating daily high-resolution PM2.5 concentrations over Texas: Machine Learning approach',
        'author': '(Ghahremanloo et al. 2021)',
        'year': 2021,
        'models': [rf, lme, linreg],
        'data': {
            'spatial resolution': [one_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2014',
                'end': 'December 2018'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': maic
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://www.sciencedirect.com/science/article/pii/S1352231021000273'
    },
    {
        'title': 'Estimating Daily PM2.5 and PM10 over Italy Using an Ensemble Model',
        'author': '(Shtein et al. 2020)',
        'year': 2020,
        'models': [lme, rf, xgboost],
        'data': {
            'spatial resolution': [one_km],
            'temporal resolution': [one_day],
            'missing': [interp],
            'training': {
                'start': 'January 2013',
                'end': 'December 2015'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': maic
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://doi.org/10.1021/acs.est.9b04279'
    },
    {
        'title': 'Estimating Ground-Level PM2.5 by Fusing Satellite and Station Observations: A Geo-Intelligent Deep Learning Approach',
        'author': '(Li et al. 2017)',
        'year': 2017,
        'models': [deepbelief],
        'data': {
            'spatial resolution': [ten_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2015',
                'end': 'Dcember 2015'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': dbdt
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1002/2017GL075710'
    },
    {
        'title': 'Estimating PM2.5 concentration of the conterminous United States via interpretable convolutional neural networks',
        'author': '(Park et al. 2020)',
        'year': 2020,
        'models': [cnn],
        'data': {
            'spatial resolution': [twelve_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2011',
                'end': 'December 2011'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': maic
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'http://www.sciencedirect.com/science/article/pii/S0269749119335341'
    },
    {
        'title': 'Estimating PM2.5 Concentrations in the Conterminous United States Using the Random Forest Approach',
        'author': '(Hu et al. 2017)',
        'year': 2017,
        'models': [rf],
        'data': {
            'spatial resolution': [twelve_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2011',
                'end': 'December 2011'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': dbdt
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://doi.org/10.1021/acs.est.7b01210'
    },
    {
        'title': 'Estimating PM2.5 with high-resolution 1-km AOD data and an improved machine learning model over Shenzhen, China',
        'author': '(Chen et al. 2020)',
        'year': 2020,
        'models': [rf],
        'data': {
            'spatial resolution': [one_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2016',
                'end': 'December 2018'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': maic
            },
            'land use': false,
            'meteorological': true
        },
        'url' : 'https://www.sciencedirect.com/science/article/pii/S0048969720346222'
    },
    {
        'title': 'Evaluation of machine learning techniques with multiple remote sensing datasets in estimating monthly concentrations of ground-level PM2.5',
        'author': '(Xu et al. 2018)',
        'year': 2018,
        'models': [linreg, svm, mars, rf, xgboost, cubist, brnn, lasso],
        'data': {
            'spatial resolution': [one_km],
            'temporal resolution': [one_month],
            'missing': [],
            'training': {
                'start': 'January 2001',
                'end': 'December 2014'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': null
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'http://www.sciencedirect.com/science/article/pii/S0269749118324229'
    },
    {
        'title': 'High Spatial Resolution PM2.5 Retrieval Using MODIS and Ground Observation Station Data Based on Ensemble Random Forest',
        'author': '(Chen et al. 2019)',
        'year': 2019,
        'models': [rf],
        'data': {
            'spatial resolution': [one_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2015',
                'end': 'December 2016'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': dbdt
            },
            'land use': false,
            'meteorological': false
        },
        'url' : 'https://doi.org/10.1109/ACCESS.2019.2908975'
    },
    {
        'title': 'Satellite-based ground PM2.5 estimation using timely structure adaptive modeling',
        'author': '(Fang et al. 2016)',
        'year': 2016,
        'models': [tsam],
        'data': {
            'spatial resolution': [ten_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'June 1, 2013',
                'end': 'May 31, 2014'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': dbdt
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'http://www.sciencedirect.com/science/article/pii/S0034425716303303'
    },
    {
        'title': 'Spatial modeling of PM2.5 concentrations with a multifactoral radial basis function neural network',
        'author': '(Zou et al. 2015)',
        'year': 2015,
        'models': [rbfnet],
        'data': {
            'spatial resolution': [point],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2006',
                'end': 'December 2006',
            },
            'testing': {
                'start': 'January 2006',
                'end': 'December 2006',
            },
            'AOD': {
                'source': null,
                'algorithm': null
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://doi.org/10.1007/s11356-015-4380-3'
    },
    {
        'title': 'Spatiotemporal prediction of continuous daily PM2.5 concentrations across China using a spatially explicit machine learning algorithm',
        'author': '(Zhan et al. 2017)',
        'year': 2017,
        'models': [gb],
        'data': {
            'spatial resolution': [point_5_by_point_5],
            'temporal resolution': [one_day],
            'missing': [true],
            'training': {
                'start': 'January 2014',
                'end': 'December 2014'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': dbdt
            },
            'land use': false,
            'meteorological': true
        },
        'url' : 'http://www.sciencedirect.com/science/article/pii/S1352231017300936'
    },
    {
        'title': 'Spatiotemporal prediction of fine particulate matter using high-resolution satellite images in the Southeastern US 2003–2011',
        'author': '(Lee et al. 2016)',
        'year': 2016,
        'models': [],
        'data': {
            'spatial resolution': [one_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2003',
                'end': 'December 2011'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': maic
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'https://www.nature.com/articles/jes201541'
    },
    {
        'title': 'Stacking machine learning model for estimating hourly PM2.5 in China based on Himawari 8 aerosol optical depth data',
        'author': '(Chen et al. 2019)',
        'year': 2019,
        'models': [adaboost, xgboost, rf],
        'data': {
            'spatial resolution': [zero_poin_zero_zero_five],
            'temporal resolution': [one_hour],
            'missing': [],
            'training': {
                'start': 'January 2016',
                'end': 'Decemer 2016'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': hima,
                'algorithm': null
            },
            'land use': true,
            'meteorological': true
        },
        'url' : 'http://www.sciencedirect.com/science/article/pii/S0048969719339981'
    },
    {
        'title': 'Synergy of satellite and ground based observations in estimation of particulate matter in eastern China',
        'author': '(Wu et al. 2012)',
        'year': 2012,
        'models': [mlp],
        'data': {
            'spatial resolution': [ten_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2007',
                'end': 'December 2008'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': dbdt
            },
            'land use': false,
            'meteorological': true
        },
        'url' : 'https://www.sciencedirect.com/science/article/pii/S0048969712008595'
    },
    {
        'title': 'The empirical relationship between the PM2.5 concentration and aerosol optical depth over the background of North China from 2009 to 2011',
        'author': '(Xin et al. 2014)',
        'year': 2014,
        'models': [linreg],
        'data': {
            'spatial resolution': [ten_km],
            'temporal resolution': [one_day],
            'missing': [],
            'training': {
                'start': 'January 2009',
                'end': 'December 2011'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': modis,
                'algorithm': null
            },
            'land use': false,
            'meteorological': false
        },
        'url' : 'http://www.sciencedirect.com/science/article/pii/S016980951300313X'
    },
    {
        'title': 'Comparison of Six Machine Learning Methods for Estimating PM2.5 Concentration Using the Himawari-8 Aerosol Optical Depth',
        'author': '(Zuo et al. 2020)',
        'year': 2020,
        'models': [bagging, decision_tree, knn, svm, rf, gb ],
        'data': {
            'spatial resolution': [five_km],
            'temporal resolution': [one_hour],
            'missing': [],
            'training': {
                'start': 'July 2015',
                'end': 'December 2017'
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': hima,
                'algorithm': null
            },
            'land use': false,
            'meteorological': false
        },
        'url' : 'https://doi.org/10.1007/s12524-020-01154-z'
    }
]
/*
    {
        'title': '',
        'author': '',
        'year': '',
        'models': [],
        'data': {
            'spatial resolution': [],
            'temporal resolution': [],
            'missing': [],
            'training': {
                'start': null,
                'end': null
            },
            'testing': {
                'start': null,
                'end': null,
            },
            'AOD': {
                'source': null,
                'algorithm': null
            },
            'land use': null,
            'meteorological': null
        },
        'url' : ''
    },
*/

fs.writeFile('bibliography.json', JSON.stringify(papers), function (err) {
  if (err) {
    console.log('Could not save the bib');
    return console.log(err);
  }
  else
  {
      console.log('Successfully saved the bib')
  }
});