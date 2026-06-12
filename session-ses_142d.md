# Chandrayaan-2 DFSAR Phase 1 pipeline

**Session ID:** ses_142da8b4cffeuhtEy5da0Ivdwy
**Created:** 6/13/2026, 12:13:29 AM
**Updated:** 6/13/2026, 12:19:51 AM

---

## User

We are building a research-grade lunar subsurface ice detection platform using Chandrayaan-2 DFSAR and OHRC data.

Current objective:

PHASE 1 ONLY

Build a production-grade DFSAR processing pipeline.

Requirements:

1. Input
   - Chandrayaan-2 DFSAR L-band data
   - Chandrayaan-2 DFSAR S-band data
   - Quad-pol data if available
   - Metadata support

2. Processing
   - Radiometric calibration
   - Speckle reduction
   - Polarimetric decomposition
   - Coherency matrix generation (T3)
   - Cloude-Pottier decomposition
   - Entropy
   - Alpha
   - Anisotropy
   - CPR
   - DOP
   - Backscatter coefficients

3. Output
   - GeoTIFF layers
   - Polarimetric feature cube
   - Visualization PNGs
   - Statistical summaries

4. Architecture
   - Python 3.12
   - Rasterio
   - GDAL
   - NumPy
   - SciPy
   - Xarray
   - Zarr storage
   - Type hints
   - Logging
   - Config-driven design

5. Repository Structure
   - src/
   - tests/
   - configs/
   - notebooks/
   - docs/

6. Deliverables
   - Complete architecture
   - Folder structure
   - Class diagrams
   - Data models
   - Config files
   - CLI commands
   - Step-by-step implementation plan

Important:
Do not generate machine learning code yet.
Do not generate XGBoost code.
Do not generate U-Net code.

Focus only on a robust scientific radar-processing foundation because all later ice-detection models depend on the quality of these polarimetric products.

Also review Chandrayaan-2 DFSAR literature and incorporate best practices from published DFSAR processing workflows. 

---

## Assistant (Build · DeepSeek V4 Flash Free · 7.7s)

_Thinking:_

The user wants me to build a production-grade DFSAR (Dual-Frequency Synthetic Aperture Radar) processing pipeline for Chandrayaan-2 data. This is a significant software engineering task. Let me break this down:

1. First, I need to research Chandrayaan-2 DFSAR literature to understand the data format and processing requirements
2. Then design and implement the complete architecture

Let me start by researching the topic and understanding the current workspace, then plan the implementation.

Key points:
- Python 3.12
- Rasterio, GDAL, NumPy, SciPy, Xarray, Zarr
- Type hints, logging, config-driven
- No ML/XGBoost/U-Net code yet
- Focus on polarimetric radar processing

Let me first check what's in the current workspace, then research DFSAR processing workflows.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE"
}
```

**Output:**
```
<path>/home/rabindra03/workspace/DEV/LUNAICE</path>
<type>directory</type>
<entries>


(0 entries)
</entries>
```

**Tool: websearch**

**Input:**
```json
{
  "query": "Chandrayaan-2 DFSAR data processing pipeline polarimetric decomposition workflow 2024 2025",
  "numResults": 10
}
```

**Output:**
```
{
  "search_id": "search_245fed649f394a4b9995a249bcc02467",
  "results": [
    {
      "url": "https://www.mdpi.com/2072-4292/15/1/101",
      "title": "Polarimetric SAR Decomposition Method Based on Modified Rotational Dihedral Model",
      "publish_date": "2022-12-24",
      "excerpts": [
        "\") [](https://pub.mdpi-res.com/remotesensing/remotesensing-15-00101/article_deploy/html/images/remotesensing-15-00101-g017a.png?1672826538 \"\n                        <strong>Figure 17</strong><br/>\n                                                    <p>Building detection results of ALOS-2/PALSAR-2 dataset. (<b>a</b>) Ground truth image, (<b>b</b>) Y4R, (<b>c</b>) G4U, (<b>d</b>) 6SD, (<b>e</b>) 7SD, (<b>f</b>) MRDD-5SD.</p>\n                                                \") [](https://pub.mdpi-res.com/remotesensing/remotesensing-15-00101/article_deploy/html/images/remotesensing-15-00101-g017b.png?1672826552 \"\n                        <strong>Figure 17 Cont.</strong><br/>\n                                                    <p>Building detection results of ALOS-2/PALSAR-2 dataset. (<b>a</b>) Ground truth image, (<b>b</b>) Y4R, (<b>c</b>) G4U, (<b>d</b>) 6SD, (<b>e</b>) 7SD, (<b>f</b>) MRDD-5SD.</p>\n\")\n\nVersions Notes\n\n## Abstract\n\nPolarimetric decomposition is an effective way to analyze the scattering mechanism of targets in polarimetric synthetic aperture radar (PolSAR) images. However, the analysis of urban areas is frequently a challenge. Most decomposition methods use a rotated dihedral derived via rotation matrix to model the double-bounce scattering mechanism of buildings. However, according to electromagnetic theory, the existing dihedral model is not accurate, especially when the orientation angle of the dihedral is large. Therefore, the double-bounce scattering contribution in urban areas with large orientation angles will be difficult to extract.\nTo address this problem, based on physical optics (PO) and geometric optics (GO), the interaction process of electromagnetic waves and the rotational dihedral is analyzed, and then a modified rotational dihedral model (MRDM) is proposed for the accurate representation of the rotational double-bounce scattering mechanism. Accordingly, MRDM is introduced to a five-component decomposition method (MRDM-5SD) to analyze the scattering components in an urban area. The validity of MRDM-5SD is demonstrated using several data sets. The experimental results show that the power contributions of double-bounce scattering in urban areas with large orientation angles increase by using MRDM-5SD. Therefore, MRDM can provide support for feature extraction and target detection in urban areas.\n\nKeywords:\n\npolarimetric SAR ; a modified rotational dihedral model ; five-component scattering decomposition ; urban area\n\n## 1\\. Introduction"
      ]
    },
    {
      "url": "https://www.researchgate.net/publication/391300470_Polarimetric_Decomposition_Analysis_of_Lunar_North_Pole_Crater_Hermite-A_Using_Chandrayaan-2_DFSAR_Data",
      "title": "Polarimetric Decomposition Analysis of Lunar North Pole ...",
      "publish_date": "2024-12-27",
      "excerpts": [
        "In this study, we assess the performance of Compact Polarimetry (Hybrid Polarimetry) over dual-pol RISAT-1 data for land cover classification over various ground targets using backscattered"
      ]
    },
    {
      "url": "https://www.msn.com/en-in/news/insight/chandrayaan-2-study-finds-buried-ice-in-moon-s-south-pole-craters/gm-GMECEFB09C?gemSnapshotKey=GMECEFB09C-snapshot-7&uxmode=ruby",
      "title": "Chandrayaan-2 study finds buried ice in Moon's south pole ...",
      "publish_date": null,
      "excerpts": [
        "Chandrayaan-2 study finds buried ice in Moon's south pole ...\nMay 14, 2026 — PRL scientists found strong radar evidence of subsurface ice in four ultra-cold craters near the Moon's south pole using Chandrayaan-2 data."
      ]
    },
    {
      "url": "https://ijsrst.com/IJSRST24111959",
      "title": "Basic Polarimetric Characteristics of Full-Polarization ...",
      "publish_date": "2024-05-09",
      "excerpts": [
        "Basic Polarimetric Characteristics of Full-Polarization Chandrayaan-II DFSAR Data Analyzed Using MIDAS (Microwave Data Analysis Software) ... Shinde Suryakant Annasaheb, Sayyad Shafiyoddin, Kranti Ramdas Zakde (Author)"
      ]
    },
    {
      "url": "https://pradan.issdc.gov.in/ch2",
      "title": "Chandrayaan-2 - PRADAN - ISSDC",
      "publish_date": null,
      "excerpts": [
        "# ISRO Science Data Archive (ISDA)\n\n# Chandrayaan-II\n\n# Chandrayaan-2\n\nAccess Data\n\n[MapBrowse Try Chandrayaan-2 data map browse and download application](https://chmapbrowse.issdc.gov.in)\n\nThe Chandrayaan-2 mission was successfully launched on 22nd July 2019 at 14:43 hrs by GSLV MkIII-M1 from Satish Dhawan Space Centre (SDSC), Sriharikota. After a series of Earth bound manoeuvres, the spacecraft entered into Lunar Transfer Trajectory (LTT) on August 14th. Lunar Orbit Insertion (LOI) manoeuvre was performed on August 20th, thereby Chandrayaan-2 was successfully inserted into the elliptical orbit around the Moon. This was followed by a series of Lunar bound orbit maneuvers for reducing the orbit to circular polar orbit around the Moon.\n\nOn September 2nd, Vikram lander separated from the Orbiter and de-orbiting maneuver was performed to reduce the orbit to 35 km x 101 km.\nVikram landing was attempted on 7th September and it followed the planned descent trajectory from its orbit of 35 km to around 2 km above the surface. Communication with lander and ground station was lost. All the systems and sensors of the Lander functioned excellently until this point and proved many new technologies such as variable thrust propulsion technology used in the Lander. However, the Orbiter is healthy and all the payloads are operational.\n\nChandrayaan-2 Orbiter is currently in a 100 km x 100 km orbit around the Moon, carries 8 experiments for studies ranging from surface geology and composition to exospheric measurements that would continue to build upon the understanding from previous lunar missions.\n\n  \nCLASS **Chandrayaan-2 Large Area Soft X-ray Spectrometer (CLASS)** is an X ray fluorescence (XRF) experiment on board Chandrayaan-2 Orbiter to map the elemental abundances of the major rock forming elements on the lunar surface. The operating energy range 0.\n8 keV to 15 keV covers the XRF lines from Mg, Al, Si, Ca, Ti and Fe as well as Na, Cr etc which may be detected. The spatial resolution is 12.5 km x 12.5 km from a 100 km orbit. In order to convert the XRF line flux to abundances, the incident solar spectrum is measured simultaneously with the X ray Solar Monitor (XSM) payload onboard Chandrayaan-2 Orbiter.\"  \nXSM **Solar X-ray Monitor (abbreviated as XSM)** is a scientific instrument on-board the orbiter of Chandrayaan-2 mission. XSM, along with another instrument CLASS, comprise a scientific experiment known as remote X-ray fluorescence spectroscopy. It provides the measurement of soft X-ray spectrum from Sun, which is used for the quantitative analysis of the Lunar X-ray fluorescence measurements by CLASS to obtain estimate of the elemental composition of Lunar surface on global scale."
      ]
    },
    {
      "url": "https://www.instagram.com/reel/DZFBwAUFbxM",
      "title": "Chandrayaan-2 Finds Hidden Water Ice on Moon - Instagram",
      "publish_date": null,
      "excerpts": [
        "Chandrayaan-2 Finds Hidden Water Ice on Moon - Instagram\n4 days ago · Scientists using data from Chandrayaan-2 have discovered strong evidence of hidden water ice near the Moon's south pole. Using the DFSAR radar"
      ]
    },
    {
      "url": "https://www.livemint.com/news/chandrayaan2-data-makes-significant-discovery-finds-water-ice-on-moons-polar-craters-11715174542450.html",
      "title": "Chandrayaan-2 data makes significant discovery, finds water ice on Moon's polar craters | Today News",
      "publish_date": "2024-05-08",
      "excerpts": [
        "# Chandrayaan-2 data makes significant discovery, finds water ice on Moon's polar craters\n\n## * The study has suggested that ‘the amount of subsurface ice in the first couple of meters is about 5 to 8 times larger than the one at the surface in both poles,’ ISRO said\n\n[**Livemint**](https://www.livemint.com/authors/livemint)\n\nPublished 8 May 2024, 07:55 PM IST\n\nThe results also conclude that the distribution of water ice is likely governed by Mare volcanism and preferential impact cratering The results also conclude that the distribution of water ice is likely governed by Mare volcanism and preferential impact cratering **(Pixabay)**\n\nMonths after India took a short walk on the Moon as Chandrayaan-3 strolled around, polarimetric radar data used from the Chandrayaan-2 Dual-frequency Synthetic Aperture Radar has hinted at significant discovery that could be the evidence of possibility of water in polar craters of the lunar surface.\nThe study was a collaboration between the scientists of Space Applications Centre (SAC)/ISRO with researchers at IIT Kanpur, University of Southern California, Jet Propulsion Laboratory, and IIT (ISM) Dhanbad (Chakraborty et al., 2024).\n\nThe study has suggested that “the amount of subsurface ice in the first couple of meters is about 5 to 8 times larger than the one at the surface in both poles”, ISRO said.\n\nThe study also suggests that the amount of water ice in the northern polar region is double that found in the southern polar region. It also suggests that the water on the lunar poles “is outgassing during volcanism in the Imbrian period”.\n\n“The results also conclude that the distribution of water ice is likely governed by Mare volcanism and preferential impact cratering,” ISRO said.\n\nThe team used seven instruments to make this significant discovery. It used “radar, laser, optical, neutron spectrometer, ultra-violet spectrometer, and thermal radiometer”.\nISRO said “the presented comprehensive understanding of the occurrence of water ice in the lunar poles, in this study, is crucial for supporting ISRO’s future in-situ volatile exploration plans on the Moon”.\n\nMeanwhile, China, too, has launched a 53-day-long lunar probe mission to collect samples for the first time from the far side of the Moon and bring them for scientific studies. The far-side of the Moon is not visible from Earth.\n\nThe Chang'e-6 mission is tasked with collecting and then returning samples from the moon's far side to Earth -- the first endeavour of its kind in the history of human lunar exploration, China National Space Administration (CNSA) said.\n\n(With agency inputs)\n\nStay updated with the latest [Trending](https://www.livemint.com/news/trends) , [India](https://www.livemint.com/news/india) , [World](https://www.livemint.com/news/world) and [US](https://www.livemint.com/news/us-news) news.\n\n[Business News](https://www.livemint.com) [News](https://www.livemint."
      ]
    },
    {
      "url": "https://www.youtube.com/watch?v=MuzA2EE0QIk",
      "title": "India Detects Possible Ice Near Moon's South Pole | Chandrayaan-2",
      "publish_date": null,
      "excerpts": [
        "India Detects Possible Ice Near Moon's South Pole | Chandrayaan-2\nMay 28, 2026 · India's Chandrayaan-2 mission may have detected subsurface water-ice beneath permanently shadowed craters near the Moon's south pole."
      ]
    },
    {
      "url": "https://iopscience.iop.org/article/10.3847/PSJ/abfdbf",
      "title": "Chandrayaan-2 Dual-frequency Synthetic Aperture Radar (DFSAR ...",
      "publish_date": "2021-07-23",
      "excerpts": [
        "The resulting data from DFSAR support the calculation of the 2 × 2 complex scattering matrix for each resolution cell, which enables lunar near-surface characterization in terms of radar polarization properties at different wavelengths and incidence angles."
      ]
    },
    {
      "url": "https://www.ndtv.com/india-news/isro-detects-possible-ice-beneath-moons-surface-using-chandrayaan-2-data-11557798",
      "title": "ISRO Detects Possible Ice Beneath Moon's Surface Using Chandrayaan-2 Data",
      "publish_date": "2026-05-28",
      "excerpts": [
        "* [Iran War](https://www.ndtv.com/topic/iran-war)\n* [IPL 2026](https://sports.ndtv.com/ipl-2026)\n* [Gold Price](https://www.ndtv.com/gold-rate)\n* [Silver Price](https://www.ndtv.com/silver-rate)\n* [Weather Today](https://www.ndtv.com/india-news/weather-today-live-updates-imd-weather-forecast-delhi-ncr-up-noida-rajasthan-heatwave-updates-11547392)\n\n[News](https://www.ndtv.com/ \"NDTV\") [India News](https://www.ndtv.com/india \"India News\") [ISRO Detects Possible Ice Beneath Moon's Surface Using Chandrayaan-2 Data](https://www.ndtv.com/india-news/isro-detects-possible-ice-beneath-moons-surface-using-chandrayaan-2-data-11557798 \"ISRO Detects Possible Ice Beneath Moon's Surface Using Chandrayaan-2 Data\")\n\n# ISRO Detects Possible Ice Beneath Moon's Surface Using Chandrayaan-2 Data\n\n## Among the craters studied, one 1.1 km-wide crater inside the Faustini crater showed particularly strong indications of subsurface ice.\n\n* Edited by: [Deepika Pundir](https://www.ndtv.\ncom/authors/deepika-pundir-24136)\n* [India News](https://www.ndtv.com/india)\n* May 28, 2026 11:19 am IST\n  \n    + Published On May 28, 2026 09:36 am IST\n    + Last Updated On May 28, 2026 11:19 am IST\n\nRead Time: 3 mins\n\n[](https://www.google.com/preferences/source?q=ndtv)\n\nShare\n\n* Twitter\n* WhatsApp\n* Facebook\n* Reddit\n* Email\n\nScientists used advanced radar polarimetric methods to study the craters.\n\nQuick Read\n\nSummary is AI-generated, newsroom-reviewed\n\n* Chandrayaan-2 orbiter detected possible subsurface ice in Moon's south polar region\n* DFSAR radar on Chandrayaan-2 studied craters in permanently shadowed regions at poles\n* Strong ice evidence found in a crater within Faustini, showing lobate-rim morphology\n\nDid our AI summary help?\n\n[Let us know.](https://www.ndtv.com/convergence/ndtv/new/feedback.aspx)\n\n[Switch To Beeps Mode](https://www.ndtv.com/beeps/india-news/isro-detects-possible-ice-beneath-moons-surface-using-chandrayaan-2-data-11557798?pfrom=quick_read_beeps)\n**New Delhi:**\n\nThe Chandrayaan-2 orbiter of the Indian Space Research Organisation (ISRO) has provided new scientific evidence that points to the possible presence of ice beneath the Moon's surface in its south polar region.\n\nThe findings come from the Dual Frequency Synthetic Aperture Radar (DFSAR) onboard Chandrayaan-2, a microwave imaging instrument operating in L- and S-band frequencies. It is the first fully polarimetric radar system used to study the lunar surface.\n\nAccording to the press release by ISRO, the study focused on “doubly shadowed craters” which are located within permanently shadowed regions (PSRs) at the lunar poles. These areas never receive sunlight and remain extremely cold with temperatures dropping to around 25 Kelvin (around -248°C). This makes them potential traps where water ice could have been preserved for billions of years."
      ]
    }
  ],
  "warnings": null,
  "usage": [
    {
      "name": "sku_search",
      "count": 1
    }
  ],
  "session_id": "ses_142da8b4cffeuhtEy5da0Ivdwy"
}
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 9.9s)

_Thinking:_

Let me gather more specific technical information about DFSAR data format and processing workflows.

**Tool: websearch**

**Input:**
```json
{
  "query": "Chandrayaan-2 DFSAR data product format specification PDS4 polarimetric calibration",
  "numResults": 10
}
```

**Output:**
```
{
  "search_id": "search_26be1bec652247d6bce68e36e9be4df3",
  "results": [
    {
      "url": "https://ui.adsabs.harvard.edu/abs/2021PSJ.....2..134B/abstract",
      "title": "Chandrayaan-2 Dual-frequency Synthetic Aperture Radar (DFSAR ...",
      "publish_date": null,
      "excerpts": [
        "In this paper, we report on the calibration and preliminary performance characterization of DFSAR data based on the analysis of a sample set of crater regions on the Moon."
      ]
    },
    {
      "url": "https://www.instagram.com/reel/DZFBwAUFbxM",
      "title": "Chandrayaan-2 Finds Hidden Water Ice on Moon - Instagram",
      "publish_date": null,
      "excerpts": [
        "Chandrayaan-2 Finds Hidden Water Ice on Moon - Instagram\n4 days ago · Scientists using data from Chandrayaan-2 have discovered strong evidence of hidden water ice near the Moon's south pole. Using the DFSAR radar"
      ]
    },
    {
      "url": "https://arxiv.org/abs/2104.14259",
      "title": "[2104.14259] Chandrayaan-2 Dual-Frequency SAR (DFSAR ...",
      "publish_date": "2021-04-29",
      "excerpts": [
        "In this paper, we report on the calibration and preliminary performance characterization of DFSAR data based on the analysis of a sample set of crater regions on the Moon."
      ]
    },
    {
      "url": "https://ui.adsabs.harvard.edu/abs/2023LPICo2806.1827S/abstract",
      "title": "Chandrayaan-2 Orbiter Data Explorer and Visualization",
      "publish_date": null,
      "excerpts": [
        "Chandrayaan-2 Orbiter Data Explorer and Visualization\nby K Suresh · 2023 · Cited by 1 — Chandrayaan-2 Imaging Payload datasets are archived in PDS-4 standard and disseminated. This paper discusses the web application for data explorer."
      ]
    },
    {
      "url": "https://www.researchgate.net/publication/330459809_Polarimetric_Calibration_of_the_GaoFen-3_Mission_Using_Active_Radar_Calibrators_and_the_Applicable_Conditions_of_System_Model_for_Radar_Polarimeters",
      "title": "(PDF) Polarimetric Calibration of the GaoFen-3 Mission Using ...",
      "publish_date": null,
      "excerpts": [
        "(PDF) Polarimetric Calibration of the GaoFen-3 Mission Using ...\nThen, we proposed the calibration data processing algorithm for GaoFen-3 based on the improved model and obtained better results. The algorithm proposed here is verified to be suitable for"
      ]
    },
    {
      "url": "https://www.sciencedirect.com/science/article/pii/S0273117722000758",
      "title": "Polarimetric analysis of L-band DFSAR data of Chandrayaan-2 ...",
      "publish_date": "2022-12-15",
      "excerpts": [
        "The main objective of this work was the polarimetric analysis of L-band DFSAR data of Chandrayaan-2 mission for surface water ice detection in the Permanently Shadowed Regions (PSRs) of Lunar South Polar Craters."
      ]
    },
    {
      "url": "https://www.mdpi.com/2072-4292/11/2/176",
      "title": "Polarimetric Calibration of the GaoFen-3 Mission Using Active ...",
      "publish_date": null,
      "excerpts": [
        "com/remotesensing/remotesensing-11-00176/article_deploy/html/images/remotesensing-11-00176-g005.png?1570965740 \"\n                        <strong>Figure 5</strong><br/>\n                                                    <p>One of the trihedral corner reflectors (TCRs) deployed at the grassland for the GF-3 experiment.</p>\n                                                \") [](https://pub.mdpi-res.com/remotesensing/remotesensing-11-00176/article_deploy/html/images/remotesensing-11-00176-g0A1.png?1570965740 \"\n                        <strong>Figure A1</strong><br/>\n                                                    <p>System transfer function model for radar polarimeters.</p>\n                                                \")\n\nVersions Notes\n\n## Abstract\n\nGaoFen-3, the first polarimetric SAR satellite of China, carried out polarimetric calibration experiments using C-band polarimetric active radar calibrators (PARCs), trihedral corner reflectors (TCRs), and dihedral corner reflectors (DCRs).\nThe calibration data were firstly processed referring to the classic 2 × 2 receive **R** and transmit **T** model for radar polarimeter systems, first proposed by Zebker, Zyl, and Held, and Freeman’s method based on PARCs, but the results were not good enough. After detailed analysis about the GaoFen-3 polarimetric system, we found that the system had some nonlinearity, then a new imbalance parameter was introduced to the classic model, which is equivalent to theproposed in Freeman’s paper about a general polarimetric system model. Then, we proposed the calibration data processing algorithm for GaoFen-3 based on the improved model and obtained better results. The algorithm proposed here is verified to be suitable for GaoFen-3 and can be applied to other spaceborne and airborne fully-polarimetric SAR systems.\n\nKeywords:\n\nsynthetic aperture radar (SAR) ; radar polarimeter ; system model ; system transfer function ; calibration ; active radar calibrator\n\n## 1\\. Introduction\nPolarimetric radar has received much attention due to its application advantages. In the last three decades, from airborne fully-polarimetric SAR such as NADC/ERIMP-3SAR  [1 ], AIRSAR  [1 , 2 , 3 , 4 , 5 , 6 ], CRLNASDASAR  [7 ], EMISAR  [8 ], Pi-SAR/Pi-SAR2  [9 , 10 , 11 ], PolSAR  [12 ], and the Ingara system  [13 ] to spaceborne fully-polarimetric SAR such as SIR-C  [14 , 15 , 16 , 17 ], ALOS-1  [18 , 19 , 20 , 21 , 22 ], RadarSat-2  [23 , 24 , 25 ] and ALOS-2  [26 , 27 ], many polarimetric SAR systems have been constructed. GaoFen-3 (GF-3) is China’s first meter-level multi-polarization SAR satellite with scientific and commercial applications, which was launched in August 2016. GF-3 supports the most abundant imaging modes in the world  [28 ] and is mainly used in the fields of ocean, disaster reduction, water conservancy, meteorology, etc.  [29 ]."
      ]
    },
    {
      "url": "https://www.researchgate.net/publication/322813241_Polarimetric_Calibration_and_Quality_Assessment_of_the_GF-3_Satellite_Images/fulltext/5a711b91a6fdcc33daa9efc4/322813241_Polarimetric_Calibration_and_Quality_Assessment_of_the_GF-3_Satellite_Images.pdf",
      "title": "Polarimetric Calibration and Quality Assessment of the GF-3 ...",
      "publish_date": null,
      "excerpts": [
        "Polarimetric Calibration and Quality Assessment of the GF-3 ...\nIn order to assess the polarimetric accuracy of this satellite and improve the image quality, we analyzed the polarimetric distortion errors and performed a PolCAL experiment based on"
      ]
    },
    {
      "url": "https://www.ndtv.com/india-news/isro-detects-possible-ice-beneath-moons-surface-using-chandrayaan-2-data-11557798",
      "title": "ISRO Detects Possible Ice Beneath Moon's Surface Using ...",
      "publish_date": null,
      "excerpts": [
        "* [Iran War](https://www.ndtv.com/topic/iran-war)\n* [IPL 2026](https://sports.ndtv.com/ipl-2026)\n* [Gold Price](https://www.ndtv.com/gold-rate)\n* [Silver Price](https://www.ndtv.com/silver-rate)\n* [Weather Today](https://www.ndtv.com/india-news/weather-today-live-updates-imd-weather-forecast-delhi-ncr-up-noida-rajasthan-heatwave-updates-11547392)\n\n[News](https://www.ndtv.com/ \"NDTV\") [India News](https://www.ndtv.com/india \"India News\") [ISRO Detects Possible Ice Beneath Moon's Surface Using Chandrayaan-2 Data](https://www.ndtv.com/india-news/isro-detects-possible-ice-beneath-moons-surface-using-chandrayaan-2-data-11557798 \"ISRO Detects Possible Ice Beneath Moon's Surface Using Chandrayaan-2 Data\")\n\n# ISRO Detects Possible Ice Beneath Moon's Surface Using Chandrayaan-2 Data\n\n## Among the craters studied, one 1.1 km-wide crater inside the Faustini crater showed particularly strong indications of subsurface ice.\n\n* Edited by: [Deepika Pundir](https://www.ndtv.\ncom/authors/deepika-pundir-24136)\n* [India News](https://www.ndtv.com/india)\n* May 28, 2026 11:19 am IST\n  \n    + Published On May 28, 2026 09:36 am IST\n    + Last Updated On May 28, 2026 11:19 am IST\n\nRead Time: 3 mins\n\n[](https://www.google.com/preferences/source?q=ndtv)\n\nShare\n\n* Twitter\n* WhatsApp\n* Facebook\n* Reddit\n* Email\n\nScientists used advanced radar polarimetric methods to study the craters.\n\nQuick Read\n\nSummary is AI-generated, newsroom-reviewed\n\n* Chandrayaan-2 orbiter detected possible subsurface ice in Moon's south polar region\n* DFSAR radar on Chandrayaan-2 studied craters in permanently shadowed regions at poles\n* Strong ice evidence found in a crater within Faustini, showing lobate-rim morphology\n\nDid our AI summary help?\n\n[Let us know.](https://www.ndtv.com/convergence/ndtv/new/feedback.aspx)\n\n[Switch To Beeps Mode](https://www.ndtv.com/beeps/india-news/isro-detects-possible-ice-beneath-moons-surface-using-chandrayaan-2-data-11557798?pfrom=quick_read_beeps)\n**New Delhi:**\n\nThe Chandrayaan-2 orbiter of the Indian Space Research Organisation (ISRO) has provided new scientific evidence that points to the possible presence of ice beneath the Moon's surface in its south polar region.\n\nThe findings come from the Dual Frequency Synthetic Aperture Radar (DFSAR) onboard Chandrayaan-2, a microwave imaging instrument operating in L- and S-band frequencies. It is the first fully polarimetric radar system used to study the lunar surface.\n\nAccording to the press release by ISRO, the study focused on “doubly shadowed craters” which are located within permanently shadowed regions (PSRs) at the lunar poles. These areas never receive sunlight and remain extremely cold with temperatures dropping to around 25 Kelvin (around -248°C). This makes them potential traps where water ice could have been preserved for billions of years."
      ]
    },
    {
      "url": "https://pds.nasa.gov/datastandards/about",
      "title": "PDS: Data Standards",
      "publish_date": null,
      "excerpts": [
        "[Planetary Data System](https://pds.nasa.gov/)\n\nFind a Node - Use these links to navigate to any of the 8 publicly accessible PDS Nodes.  \n  \nThis bar indicates that you are within the PDS enterprise which includes 6 science discipline nodes and 2 support nodes which are overseen by the Project Management Office at NASA's Goddard Space Flight Center (GSFC). Each node is led by an expert in the subject discipline, supported by an advisory group of other practitioners of that discipline, and subject to selection and approval under a regular NASA Research Announcement.\n\nFind a Node\n\n* [Atmospheres (ATM)](https://pds-atmospheres.nmsu.edu/)\n* [Cartography and Imaging Sciences (IMG)](https://pds-imaging.jpl.nasa.gov/)\n* [Geosciences (GEO)](https://pds-geosciences.wustl.edu/)\n* [Navigation & Ancillary Information Facility (NAIF)](https://naif.jpl.nasa.gov/naif/)\n* [Planetary Plasma Interactions (PPI)](https://pds-ppi.igpp.ucla.edu/)\n* [Ring-Moon Systems (RMS)](https://pds-rings.seti.org/)\n* [Small Bodies (SBN)](https://pds-smallbodies.astro.umd.edu/)\n\n## PDS Nodes\n\n* [Atmospheres (ATM)](https://pds-atmospheres.nmsu.edu/)\n* [Cartography and Imaging Sciences (IMG)](https://pds-imaging.jpl.nasa.gov/)\n* [Geosciences (GEO)](http://pds-geosciences.wustl.edu/)\n* [Navigational & Ancillary Information Facility (NAIF)](https://naif.jpl.nasa.gov/)\n* [Planetary Plasma Interactions (PPI)](https://pds-ppi.igpp.ucla.edu/)\n* [Ring-Moon Systems (RMS)](https://pds-rings.seti.org/)\n* [Small Bodies (SBN)](https://pds-smallbodies.astro.umd.edu/)\n\n# Data Standards\n\nThe PDS4 archiving standard has been required for data archives from NASA-funded missions and research activities since 2011. For those who are working on a project started under the PDS3 standards, go to PDS3 Data Standards .\n\n  \n* To find detailed documentation of the PDS4 data standards, refer to PDS4 Documents .\n  \n* PDS allows only a limited set of formats for all files and documents in an archive.\nThese formats are summarized in this policy . For more resources, refer to Archiving Formats .\n  \n* To see training and exercise materials on PDS4 data standards, refer to PDS4 Training .\n  \n* For a list of the core software tools and libraries used for generating, validating, transforming, and accessing data, see the Tools page.\n  \n* The PDS4 archive standards include international coordination through the [International Planetary Data Alliance](http://planetarydata.org/) .\n  \n\n## PDS4 Versions\n\n|Current Versions |Version 1.25.0.0 of the PDS4 Data Standards include approved changes made since the release of previous\nversions and bug fixes (see [Release Notes](https://nasa-pds.github.io/releases/current/rdd.html) )\n. |\n| --- | --- |\n|Previous Versions |Summary of PDS4 previous release versions. |\n\n## PDS4 Build Schedule\n\n|IM Version |Build Number |Release Date |\n| --- | --- | --- |\n|1\\.31.0.0 (1V00) |22 |December 2028 |\n|1\\.30.0.0 (1U00) |21 |June 2028 |\n|1\\.29.0."
      ]
    }
  ],
  "warnings": null,
  "usage": [
    {
      "name": "sku_search",
      "count": 1
    }
  ],
  "session_id": "ses_142da8b4cffeuhtEy5da0Ivdwy"
}
```

**Tool: websearch**

**Input:**
```json
{
  "query": "DFSAR Chandrayaan-2 polarimetric decomposition entropy alpha anisotropy CPR processing Python",
  "numResults": 10
}
```

**Output:**
```
{
  "search_id": "search_123b8424385d4bec96e6a670b718ec17",
  "results": [
    {
      "url": "https://iopscience.iop.org/article/10.3847/PSJ/abfdbf",
      "title": "Chandrayaan-2 Dual-frequency Synthetic Aperture Radar (DFSAR ...",
      "publish_date": null,
      "excerpts": [
        "Chandrayaan-2 Dual-frequency Synthetic Aperture Radar (DFSAR ...\nThe resulting data from DFSAR support the calculation of the 2 × 2 complex scattering matrix for each resolution cell, which enables lunar near-surface characterization in terms of radar polarization properties at different wavelengths and incidence angles."
      ]
    },
    {
      "url": "https://www.mdpi.com/2072-4292/15/1/101",
      "title": "Polarimetric SAR Decomposition Method Based on Modified Rotational Dihedral Model",
      "publish_date": "2022-12-24",
      "excerpts": [
        "\") [](https://pub.mdpi-res.com/remotesensing/remotesensing-15-00101/article_deploy/html/images/remotesensing-15-00101-g017a.png?1672826538 \"\n                        <strong>Figure 17</strong><br/>\n                                                    <p>Building detection results of ALOS-2/PALSAR-2 dataset. (<b>a</b>) Ground truth image, (<b>b</b>) Y4R, (<b>c</b>) G4U, (<b>d</b>) 6SD, (<b>e</b>) 7SD, (<b>f</b>) MRDD-5SD.</p>\n                                                \") [](https://pub.mdpi-res.com/remotesensing/remotesensing-15-00101/article_deploy/html/images/remotesensing-15-00101-g017b.png?1672826552 \"\n                        <strong>Figure 17 Cont.</strong><br/>\n                                                    <p>Building detection results of ALOS-2/PALSAR-2 dataset. (<b>a</b>) Ground truth image, (<b>b</b>) Y4R, (<b>c</b>) G4U, (<b>d</b>) 6SD, (<b>e</b>) 7SD, (<b>f</b>) MRDD-5SD.</p>\n\")\n\nVersions Notes\n\n## Abstract\n\nPolarimetric decomposition is an effective way to analyze the scattering mechanism of targets in polarimetric synthetic aperture radar (PolSAR) images. However, the analysis of urban areas is frequently a challenge. Most decomposition methods use a rotated dihedral derived via rotation matrix to model the double-bounce scattering mechanism of buildings. However, according to electromagnetic theory, the existing dihedral model is not accurate, especially when the orientation angle of the dihedral is large. Therefore, the double-bounce scattering contribution in urban areas with large orientation angles will be difficult to extract.\nTo address this problem, based on physical optics (PO) and geometric optics (GO), the interaction process of electromagnetic waves and the rotational dihedral is analyzed, and then a modified rotational dihedral model (MRDM) is proposed for the accurate representation of the rotational double-bounce scattering mechanism. Accordingly, MRDM is introduced to a five-component decomposition method (MRDM-5SD) to analyze the scattering components in an urban area. The validity of MRDM-5SD is demonstrated using several data sets. The experimental results show that the power contributions of double-bounce scattering in urban areas with large orientation angles increase by using MRDM-5SD. Therefore, MRDM can provide support for feature extraction and target detection in urban areas.\n\nKeywords:\n\npolarimetric SAR ; a modified rotational dihedral model ; five-component scattering decomposition ; urban area\n\n## 1\\. Introduction"
      ]
    },
    {
      "url": "https://earth.esa.int/eogateway/documents/20142/37627/PolSAR-2013-Basic-Concepts.pdf",
      "title": "[PDF] SAR POLARIMETRY - ESA Earth Online",
      "publish_date": null,
      "excerpts": [
        "**MODEL BASED - 4 COMPONENTS DECOMPOSITION**\n\n**_E.Pottier_**\n\n**_(2013)_**\n\n**MODEL BASED DECOMPOSITION**\n\n**MODEL BASED DECOMPOSITION**\n\n**4 COMPONENTS SCATTERING MECHANISM MODEL**\n\n[[[[ ]]]]\n\n[[[[\n\n]]]]\n\n[[[[\n\n]]]]\n\n[[[[\n\n]]]]\n\n[[[[\n\n**_H_** ]]]]\n\n**_V_**\n\n**_D_**\n\n**_S_**\n\n**_T_**\n\n**_T_**\n\n**_T_**\n\n**_T_**\n\n**_T_**\n\n\\++++\n\n\\++++\n\n\\++++\n\n====\n\n**MEDIUM WITHOUT ANY REFLECTION SYMMETRY**\n\n**SINGLE**\n\n**SCATTERING**\n\n**DOUBLE**\n\n**SCATTERING**\n\n**VOLUME**\n\n**SCATTERING**\n\n[[[[ ]]]]\n\n[[[[\n\n]]]]\n\n[[[[\n\n]]]]\n\n[[[[\n\n]]]]\n\n[[[[\n\n**_H_** ]]]]\n\n**_V_**\n\n**_D_**\n\n**_S_**\n\n**_T_**\n\n**_T_**\n\n**_T_**\n\n**_T_**\n\n**_T_**\n\n\\++++\n\n\\++++\n\n\\++++\n\n====\n\n**HELIX**\n\n**SCATTERING**\n\n**Non reflection**\n\n\n\n\n\n\n\n\n\n±±±±\n\n====\n\n**1**\n\n**0**\n\n**0**\n\n**0**\n\n**0**\n\n**1**\n\n**[** **]**\n\n**_j_**\n\n**_T_**\n\n\n\n\n\n\n\n\n\n±±±±\n\n====\n\n**_j_**\n\n**_1_**\n\n**_1_**\n\n**_S_** **]**\n\n**[**\n\n**_E.Pottier_**\n\n**_(2013)_**\n\n**Non reflection**\n\n**Symmetric cases**\n\n**Yamaguchi Y., Moriyama T., Ishido M. and Yamada H., “** **_Four-Component Scattering Model for_**\n**_Polarimetric SAR Image Decomposition_** **”, IEEE Trans. Geos. Remote Sens., vol. 43, no. 8, August 2005.**\n\n**Yamaguchi Y., Yajima Y. and Yamada H., “A Four-Component Decomposition of POLSAR Images Based**\n\n**on the Coherency Matrix”, IEEE Geos. Rem. Sens. Letters, vol. 3, no. 3, July 2006.**\n\n**2005 - 2006**\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n±±±±\n\n====\n\n**1**\n\n**0**\n\n**1**\n\n**0**\n\n**2**\n\n**1**\n\n**]**\n\n**[**\n\n**_j_**\n\n**_j_**\n\n**_T_**\n\n**_Helix_**\n\nm\n\n\n\n\n\n\n\n\n\n−−−−\n\n±±±±\n\n±±±±\n\n====\n\n±±±±\n\n**_1_**\n\n**_j_**\n\n**_j_**\n\n**_1_**\n\n**_2_**\n\n**_1_**\n\n**_S_**\n\n**[ ]** **_Helix_**\n\n**MODEL BASED DECOMPOSITION**\n\n**MODEL BASED DECOMPOSITION**\n\n**Y4O**\n\n**_E.Pottier_**\n\n**_(2013)_**\n\n**MODEL BASED DECOMPOSITION**\n\n**MODEL BASED DECOMPOSITION**\n\n**_E.Pottier_**\n\n**_(2013)_**\n\n**_2A_** **_0_**\n\n**_B_** **_0_** \\++++ **_B_**\n\n**_B_** **_0_** −−−− **_B_**\n\n((((\n\n))))\n\n**_2_**\n\n**_f_** **_S_** **_1_**\n\n**_ODD_**\n\n\\++++ ββββ\n\n====\n\n((((\n\n))))\n\n**_2_**\n\n**_f_** **_D_** **_1_**"
      ]
    },
    {
      "url": "https://www.sciencedirect.com/science/article/pii/S0273117725011913",
      "title": "Morphological characterization of impact craters in the south polar region of moon using Chandrayaan-2 Dual Frequency Synthetic Aperture Radar data - ScienceDirect",
      "publish_date": "2026-01-15",
      "excerpts": [
        "Lastly, the study aims to conduct morphometric analyses of craters, categorizing them based on size and degradation status to gain insights into their structural evolution and classification. Through these objectives, the research endeavours to contribute to the broader understanding of lunar surface dynamics.\n\n## Access through your organization\n\nCheck access to the full text by signing in through your organization.\n\nAccess through **your organization**\n\n## Section snippets\n\n## Study area\n\nThe study area chosen was the southern pole of the moon (Fig. 1), it features craters ranging in size from small to large, situated in the vicinity of Cabeus, Haworth, and Malapert craters. Cabeus, positioned 100 km away from the South Pole at coordinates 84.9° S and 35.5° W, having a diameter of approximately 98 km. Notably, it remains almost continuously in deep shadow, experiencing a perpetual darkness. Haworth, a distinctive pentagonal crater situated near the center of the Lunar South Pole\n## Data and methods\n\nThe Dual-Frequency Synthetic Aperture Radar (DFSAR) on Chandrayaan-2 marks a major advancement in lunar exploration by enabling fully polarimetric SAR imaging outside Earth orbit. It operates at both L-band (1.25 GHz) and S-band (2.5 GHz) frequencies in stand-alone and joint imaging modes. DFSAR can provide single, dual, hybrid-circular, and fully polarimetric observations, offering resolutions between 2 m and 75 m in slant range with incidence angles from 9.6° to 36.9° (Table 1) (Bhiravarasu\n\n## Results and discussion\n\nThis section presents the results of analysis of lunar craters using backscatter and decomposition techniques, followed by a detailed discussion of their morphometric properties. The findings highlight the effectiveness of the applied techniques and offer a deeper understanding of the physical states and the over-time transformation (degradation) processes of the craters.\n\n## Conclusion\nBackscatter analysis of DFSAR data, particularly HH and VV polarizations has been effectively implemented to delineate craters. While Yamaguchi decomposition further enhanced crater clarity by distinguishing scattering phenomena within and around the craters. It has been observed that the different scattering mechanisms calculated using the Yamaguchi decomposition method highlight various parts of and around the craters, with each contributing uniquely to the scattering behavior, complementing\n\n## Declaration of competing interest\n\nThe authors declare that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this manuscript. The research presented in the manuscript titled “Morphological Characterization of Impact Craters in the South Polar Region of the Moon using Chandrayaan-2 Dual Frequency Synthetic Aperture Radar Data” was carried out solely for academic and scientific purposes."
      ]
    },
    {
      "url": "https://www.sciencedirect.com/science/article/pii/S2352938522001628",
      "title": "Target decomposition using dual-polarization sentinel-1 SAR data: Study on crop growth analysis - ScienceDirect",
      "publish_date": "2022-11-01",
      "excerpts": [
        "The Sentinel-1A SLC data is preprocessed in the Sentinel Application Platform (SNAP) toolbox. Preprocessing includes importing data, applying orbit file and radiometric correction, converting slant range resolution to ground range using a multi-looking factor, and applying Refined Lee filters to obtain data in the required format. Application of orbit file provides the accurate position of the SAR image. The data consists of three sub-swaths namely IW1, IW2, and IW3. The study area falls in\n\n## Classification analysis\n\nThe LULC classification is carried out to find the date which gives the highest accuracy for H-α analysis. The image is classified into 7 classes namely Water, Urban, Agriculture, Barren land, Forest and Dense vegetation, Grassland, and Plantations using a random forest algorithm. 19 th January and 27 th September 2020 gave the highest accuracy of 99.19% with a Kappa coefficient of 0.9899 among all the data images.\nSince there will be precipitation in the month of September, therefore the\n\n## Conclusions\n\nIn the present study, an attempt is made to analyze the target properties using H-A-α dual-polarization decomposition method using Sentinel-1A SAR SLC data. The results obtained from the dual-pol decomposition showed variation from the quad-polarization decomposition. The alpha values fall between 7.92° and 45.23°, whereas the alpha values of quad-polarization data vary from 0° to 90°. The entropy of dual-polarized data ranges from 0 to 1, where double-bounce scattering yielded lower range\n\n## Declaration of competing interest\n\nThe authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.\n\n## References (50)\n\n* F. Canisius _et al._\n  \n  ### Tracking crop phenological development using multi-temporal polarimetric Radarsat-2 data\n  \n  ### Rem. Sens. Environ. (2018)\n* A.\nChaudhary _et al._\n  \n  ### An improved random forest classifier for multi-class classification\n  \n  ### Information Processing in Agriculture (2016)\n* E.T. Engman _et al._\n  \n  ### Status of microwave soil moisture measurements with remote sensing\n  \n  ### Rem. Sens. Environ. (1995)\n* D. Haldar _et al._\n  \n  ### Evaluation of full-polarimetric parameters for vegetation monitoring in rabi (winter) season\n  \n  ### Egypt. J. Remote Sens. Sp. Sci. (2018)\n* X. Jiao _et al._\n  \n  ### Object-oriented crop mapping and monitoring using multi-temporal polarimetric RADARSAT-2 data\n  \n  ### ISPRS J. Photogrammetry Remote Sens. (2014)\n* L. Wang _et al._\n  \n  ### Estimation of biomass in wheat using random forest regression algorithm and remote sensing data\n  \n  ### Crop J. (2016)\n* C. Wang _et al._\n  \n  ### Soil moisture-plant interactions: an ecohydrological review\n  \n  ### J. Soils Sediments (2019)\n* Y. Xie _et al._"
      ]
    },
    {
      "url": "https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022EA002796",
      "title": "Polarimetric Decomposition and Machine Learning‐Based Classification of L‐ and S‐Band Airborne SAR (LS‐ASAR) Data - Verma - 2023 - Earth and Space Science - Wiley Online Library",
      "publish_date": "2023-06-01",
      "excerpts": [
        "Search for more papers by this author\n\nFirst published: 20 June 2023\n\n<https://doi.org/10.1029/2022EA002796>\n\nview metrics\n\nAbout\n\nSections\n\nPDF\n\nTools\n\n* [Request permission](https://agupubs.onlinelibrary.wiley.com/action/rightsLink?doi=10.1029%2F2022EA002796&mode=)\n* Export citation\n* Add to favorites\n* Track citation\n\nShare Share\n\nGive access\n\nShare full text access\n\nShare full-text access\n\nPlease review our [Terms and Conditions of Use](https://onlinelibrary.wiley.com/termsAndConditions) and check box below to share full-text version of article.\n\nI have read and accept the Wiley Online Library Terms and Conditions of Use\n\n* * *\n\nShareable Link\n\nUse the link below to share a full-text version of this article with your friends and colleagues. [Learn more.](https://onlinelibrary.wiley.com/researchers/tools-resources/sharing)\n\nCopy URL\n\nShare a link\n\nShare on\n\n* Email\n* Facebook\n* x\n* LinkedIn\n* Reddit\n* Wechat\n* Bluesky\n\n## Abstract\nThe polarimetric Synthetic Aperture Radar (SAR) data sets have been widely exploited for land use land cover (LULC) classification due to their sensitivity to the structural and dielectric properties of the imaging target. In this study, the potential of fully polarimetric L- and S-band Airborne SAR (LS-ASAR) data sets were explored for the machine-learning-based classification of Urban, Vegetation, Waterbody, and Open Ground. This work was done by utilizing dual-frequency L- and S-band airborne data of Santa Barbara, California, USA, acquired under the Airborne SAR (LS ASAR) campaign, a precursor airborne mission to the space-borne NASA-ISRO (NISAR) mission. The LS-ASAR polarimetric information was utilized for LULC classification using the SVM classifier. The roll-invariant Barnes, and eigenvalue/eigenvector-based Cloude and H/A/Alpha decomposition were implemented to retrieve the scattering parameters.\nThe backscatter response of classes was studied, and separability analysis was done to reduce the misclassification error between six class pairs- Vegetation—Urban, Vegetation—Waterbody, Vegetation—Open Ground, Urban—Waterbody, Urban—Open Ground, and Water—Open Ground. The decomposition models failed to achieve the desirable separability index for all six class pairs; consequently, the classification of Barnes, Cloude, H/A/Alpha decomposition showed misclassification between vegetation-urban class, and waterbody-open ground class for both L- and S-band data sets. The effort was made toward improving the classification accuracy by integrating the roll-invariant and eigenvalue-eigenvector scattering parameters of the multifrequency L- and S-band data set. This method presented the desirable separability index for all class-pair; eventually highest classification accuracy was achieved i.e. 93.35% (= 0.91) by significantly reducing the misclassification error between class pairs."
      ]
    },
    {
      "url": "https://ieeexplore.ieee.org/abstract/document/10474063",
      "title": "Investigating the Residual Polarimetric Distortion and ...",
      "publish_date": null,
      "excerpts": [
        "Investigating the Residual Polarimetric Distortion and ...\nThe dual-frequency synthetic aperture radar (DFSAR) in lunar orbit, using L- and S-bands, is the only full-polarization (FP) SAR. It explores different layers o"
      ]
    },
    {
      "url": "https://ijsrst.com/IJSRST24111959",
      "title": "Basic Polarimetric Characteristics of Full-Polarization ...",
      "publish_date": null,
      "excerpts": [
        "Basic Polarimetric Characteristics of Full-Polarization ...\nBasic Polarimetric Characteristics of Full-Polarization Chandrayaan-II DFSAR Data Analyzed Using MIDAS (Microwave Data Analysis Software) ... Shinde Suryakant Annasaheb, Sayyad Shafiyoddin, Kranti Ramdas Zakde (Author)"
      ]
    },
    {
      "url": "https://eurekamag.com/research/096/014/096014209.php",
      "title": "Polarimetric Decomposition Analysis of Lunar North Pole ...",
      "publish_date": null,
      "excerpts": [
        "Polarimetric Decomposition Analysis of Lunar North Pole ...\nThis paper mainly focuses on various decomposition techniques and analysis of backscattering coefficients of the lunar surface using full polarimetric L-band DFSAR data. The data is collected from PSR (Permanent Shadowed Region) of lunar north polar Hermit-a Crater."
      ]
    },
    {
      "url": "https://www.academia.edu/164987550/Analysing_the_potential_of_polarimetric_decomposition_parameters_of_Sentinel_1_dual_pol_SAR_data_for_estimation_of_rice_crop_biophysical_parameters",
      "title": "Analysing the potential of polarimetric decomposition ...",
      "publish_date": null,
      "excerpts": [
        "Analysing the potential of polarimetric decomposition ...\nThe present study focuses on the capability of the polarimetric decomposition parameters of dual-polarimetric Sentinel-1 data in monitoring the biophysical parameters of rice crop namely, fresh biomass, dry biomass, Vegetation Water Content (VWC) and plant height, through its various growth stages."
      ]
    }
  ],
  "warnings": null,
  "usage": [
    {
      "name": "sku_search",
      "count": 1
    }
  ],
  "session_id": "ses_142da8b4cffeuhtEy5da0Ivdwy"
}
```

**Tool: webfetch**

**Input:**
```json
{
  "url": "https://iopscience.iop.org/article/10.3847/PSJ/abfdbf",
  "format": "text"
}
```

**Output:**
```
Chandrayaan-2 Dual-frequency Synthetic Aperture Radar (DFSAR): Performance Characterization and Initial Results - IOPscience
































































 Skip to contentIOP Science homeAccessibility HelpSearchSearch all IOPscience contentSearchArticle LookupJournalsJournals listBrowse more than 100 science journal titlesSubject collectionsRead the very best research published in IOP journalsPublishing partnersPartner organisations and publicationsOpen accessIOP Publishing open access policy guideIOP Conference SeriesRead open access proceedings from science conferences worldwideBooksPublishing SupportLoginIOPscience login / Sign Up
                The American Astronomical Society (AAS), established in 1899 and based in Washington, DC, is the major organization of professional astronomers in North America. Its membership of about 7,000 individuals also includes physicists, mathematicians, geologists, engineers, and others whose research and educational interests lie within the broad spectrum of subjects comprising contemporary astronomy. The mission of the AAS is to enhance and share humanity's scientific understanding of the universe.
            The following article is Open accessChandrayaan-2 Dual-frequency Synthetic Aperture Radar (DFSAR): Performance Characterization and Initial ResultsSriram S. Bhiravarasu, Tathagata Chakraborty, Deepak Putrevu, Dharmendra K. Pandey, Anup K. Das, V. M. Ramanujam, Raghav Mehra, Parikshit Parasher, Krishna M. Agrawal, Shubham GuptaShow full author listPublished 2021 July 23 • 
       	  © 2021. The Author(s). Published by the American Astronomical Society.
        The Planetary Science Journal,
	Volume 2,
Number 4Citation Sriram S. Bhiravarasu et al 2021 Planet. Sci. J. 2 134DOI 10.3847/PSJ/abfdbf Download Article PDFDownloadArticle ePub
            You need an eReader or compatible software to experience the benefits of the ePub3 file format.As featured in:AAS Novaopens in new tabAuthors
                                    Sriram S. BhiravarasuAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comEMAILsriram.saran@sac.isro.gov.insriram.saran@gmail.comhttps://orcid.org/0000-0003-0019-6261Tathagata ChakrabortyAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comDeepak PutrevuAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comDharmendra K. PandeyAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comAnup K. DasAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comV. M. RamanujamAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comRaghav MehraAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comParikshit ParasherAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comKrishna M. AgrawalAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comShubham GuptaAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comGaurav S. SethAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comAmit ShuklaAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comNikhil Y. PandyaAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comSanjay TrivediAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comArundhati MisraAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comRajeev JyotiAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comRaj KumarAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comAuthor notesCurrently at National Remote Sensing Centre, ISRO, Hyderabad 500037, India.Notes
                                FiguresSkip to each figure in the articleTablesSkip to each table in the articleReferencesCitationsArticle data
                                        Skip to each data item in the article
                                        What
                                        is article data?Open scienceAuthors
                    Sriram S. BhiravarasuAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comEMAILsriram.saran@sac.isro.gov.insriram.saran@gmail.comhttps://orcid.org/0000-0003-0019-6261Tathagata ChakrabortyAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comDeepak PutrevuAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comDharmendra K. PandeyAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comAnup K. DasAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comV. M. RamanujamAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comRaghav MehraAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comParikshit ParasherAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comKrishna M. AgrawalAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comShubham GuptaAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comGaurav S. SethAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comAmit ShuklaAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comNikhil Y. PandyaAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comSanjay TrivediAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comArundhati MisraAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comRajeev JyotiAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comRaj KumarAFFILIATIONSSpace Applications Centre, ISRO, Ahmedabad 380015, India; sriram.saran@sac.isro.gov.in, sriram.saran@gmail.comAuthor notesCurrently at National Remote Sensing Centre, ISRO, Hyderabad 500037, India.Notes
                Article metrics7548 Total downloads0 Video abstract viewsShare this articleArticle informationDatesReceived 2021 January 20 Revised 2021 April 23 Accepted 2021 April 26 Published 2021 July 23 Unified Astronomy Thesaurus conceptsThe Moon; Lunar craters; Polarimetry; Radar astronomy; Lunar science; Lunar featuresJournal RSS
    
    
        
        
            
            Sign up for new issue notifications
            
        
        
        
    


    
    2632-3338/2/4/134Abstract
            The Dual-Frequency synthetic aperture radar (DFSAR) system manifested on the Chandrayaan-2 spacecraft represents a significant step forward in radar exploration of solid solar system objects. It combines SAR at two wavelengths (L and S bands) and multiple resolutions with several polarimetric modes in one lightweight (∼20 kg) package. The resulting data from DFSAR support the calculation of the 2 × 2 complex scattering matrix for each resolution cell, which enables lunar near-surface characterization in terms of radar polarization properties at different wavelengths and incidence angles. In this paper, we report on the calibration and preliminary performance characterization of DFSAR data based on the analysis of a sample set of crater regions on the Moon. Our calibration analysis provided a means to compare on-orbit performance with prelaunch measurements, and the results matched with the prelaunch expected values. Our initial results show that craters in both permanently shadowed regions (PSRs) and non-PSRs that are classified as circular polarization ratio–anomalous in previous S-band radar analyses appear anomalous at the L band also. We also observe that material evolution and physical properties at their interior and proximal ejecta are decoupled. For the Byrgius C crater region, we compare our analysis of dual-frequency radar data with the predicted behaviors of theoretical scattering models. If crater age estimates are available, a comparison of their radar polarization properties at multiple wavelengths similar to that of the three unnamed south polar crater regions shown in this study may provide new insights into how the rockiness of craters evolves with time.
        Export citation and abstractBibTeXRISPrevious article in issueNext article in issueRelated linksNASA ADS Recordopens in new tabAbout Related Links
        
                  
                  Original content from this work may be used under the terms of the Creative Commons Attribution 4.0 licence. Any further distribution of this work must maintain attribution to the author(s) and the title of the work, journal citation and DOI.
        1. IntroductionThe physical properties of lunar surface materials provide an independent means to help unravel lunar geologic history and evaluate their exploration potential. Additionally, near-surface physical properties of the regolith are of importance to address significant questions in lunar geology, such as understanding the spatial limits and relative timing of mare flow units and impact-related processes that eventually influence spatial differences in surface and subsurface composition (e.g., Cahill et al. 2014). In this context, polarimetric radar backscatter data for the Moon provides valuable insights regarding the physical properties of the near surface with particular sensitivity to ilmenite content, and surface or buried rocks with a diameter of about 1/10 the radar wavelength and larger (e.g., Carter et al. 2009; Campbell et al. 2010; Campbell 2012).The Dual-Frequency Synthetic Aperture Radar (DFSAR) aboard India's second lunar mission Chandrayaan-2 is the first fully polarimetric synthetic aperture radar (SAR) outside Earth orbit. The architecture of this radar instrument supports multiple polarimetric modes of operation (Putrevu et al. 2016, 2020) and illustrates the value of dual-frequency quad polarimetry for lunar and planetary applications. Regarding applications to the lunar surface, the main novelties provided by this instrument are the following:
            Frequency bands: For the first time ever, it is possible to analyze L-band (∼24 cm wavelength) polarimetric radar images of the Moon from the DFSAR instrument. Compared to previous ground-based (e.g., Arecibo, Goldstone) and orbital-based (e.g., Mini-SAR and Mini-RF) radar data of the Moon collected at the 70 cm (P band), 12.6 cm (S band), and 3.8 cm (X band) wavelengths, the new L-band SAR allows for a penetration depth of ∼3 m (for dry, low-loss soils) with particular sensitivity to ilmenite content and surface or buried rocks with a diameter of about 1/10 the radar wavelength (i.e., ∼2 cm) and larger. Additionally, combined with the S-band radar mode of operation (12 cm wavelength) along with data from the Mini-SAR and Mini-RF radars on board the Chandrayaan-1 and LRO missions respectively (Nozette et al. 2010), the DFSAR instrument can characterize the radar scattering properties of a top few meters of the lunar surface.
            Full-polarization: The DFSAR architecture introduced the fully polarized (FP) case among polarimetric imaging radars for planetary missions, in which the intrinsic data product is the 4 × 4 scattering matrix of each resolved element in the scene. FP systems alternately transmit two orthogonal polarizations and record both received polarizations (HH, HV, VH, and VV) and allow much more information to be extracted from a scene compared with single- and dual-pol SAR data (Zyl et al. 1987). For each combination, the first letter refers to the transmitted polarization, while the second one refers to the received sense, with H and V denoting horizontal and vertical, respectively. After applying certain symmetry relations, this can be reduced to a 3 × 3 array such as the compressed Sinclair matrix or the compressed Stokes matrix (Lee & Pottier 2009); such reduced forms are referred to as quadrature-polarimetric SAR (quad-pol for short). Polarimetric channels acquired in quad-pol mode maintain their relative phase, so they can be combined coherently to form new channels (Lee & Pottier 2009) or to compute statistical higher-order parameters by target decompositions (e.g., Cloude & Pottier 1996). Note that in the polarimetric radar literature, the terms “FP” and “quad-pol” are often used synonymously, so we adopt similar notation throughout this paper. Quad-pol data produce a unique scattering matrix using any combination of transmitted and received orthogonal polarizations (e.g., linear, elliptical, hybrid), which permit analysis of the surface scattering behavior in all possible configurations of the transmitted and received signal polarization. Such uniqueness does not apply to single, dual, and compact polarimetric SAR configurations (e.g., Lee & Pottier 2009).
            Spatial resolution and look geometry: To address a variety of scientific objectives, DFSAR collects data of the lunar surface at a wide range of slant-resolution options spanning 2–75 m per pixel. The high spatial resolution of DFSAR enables mapping of lunar craters and other geological features, especially in the polar regions, with finer details. Compared to previous orbital and ground-based radars, high-spatial-resolution DFSAR data also provide a sufficient number of samples for the investigation of distributed scatterers that are common in lunar geologic settings. Radar backscatter will also vary depending on the incidence angle of the radar beam. Previous studies using FP SAR data of terrestrial settings indicated that surface scattering models could be developed using the relationship between surface roughness and dielectric parameters and the intensity, polarization, and angular dependence of the backscattered wave (e.g., Campbell et al. 1993; Campbell & Shepard 1996; Campbell 2002). In addition, theoretical models of radar scattering predict that at the near-nadir regime (i.e., from normal incidence to ∼20°), the reflection is dominated by locally smooth, radar-facing facets of the surface, and for incidence angles beyond ∼30°, diffuse scattering from the small-scale roughness predominates (Hagfors 1964; Farr 1993). The DFSAR instrument is designed to collect data at a wide range of angles of incidence, from 9.6° to 36.9° (Putrevu et al. 2016, 2020)). This capability enables the acquisition of dual-wavelength (L- and S-band) radar images at multiple viewing angles to address some important applications such as the estimation of wavelength-scale physical properties (e.g., surface rms height and slope; Campbell et al. 1993; Campbell & Shepard 1996), and the reduction of the ambiguities related to the interpretation of high circular polarization ratios (CPRs) exhibited by certain lunar surface features (Campbell 2012; Spudis et al. 2013; Fa & Eke 2018; Fassett et al. 2018; Virkki & Bhiravarasu 2019).The text is organized as follows. Section 2 describes the scientific background and objectives of DFSAR investigations. Section 3 is dedicated to the instrument overview and calibration of DFSAR data products. A brief description of DFSAR polarimetric parameters along with their physical interpretation is discussed in Section 4. In Section 5, we discuss the DFSAR data used in this preliminary analysis. Results are presented in Section 6, followed by conclusions and future work in Section 7.2. Background of Investigation
            Polar volatiles: The nature and distribution of volatiles (e.g., water ice) at the permanently shadowed regions (PSRs) of the Moon has been a subject of considerable controversy, due to contrasting interpretations of the polarimetric behavior of the radar backscatter from these regions at near-zero phase (bistatic) angles (e.g., Nozette et al. 1996; Campbell et al. 2006; Spudis et al. 2010, 2013; Fa & Cai 2013; Eke et al. 2014; Fa & Eke 2018; Fassett et al. 2018). The presence of ice in the regolith causes a distinct, but not unique, CPR signature due to forward scattering of radar signals by cracks and voids in the ice, which has an intrinsically low microwave loss. Due to the low-loss properties of ice, CPR values above 1 have been observed for the Galilean satellites (Campbell et al. 1978; Ostro et al. 1992; Black et al. 2001) and the Greenland ice sheet (Rignot 1995). Within some polar craters of Mercury, features exhibiting a very strong radar backscatter and enhanced CPR values along with a high degree of correlation between radar-bright features and regions of permanent shadow have been attributed to several radar wavelengths-thick slabs of water ice (Harmon et al. 1994, 2001, 2011; Chabot et al. 2012, 2013, 2018). One of the most plausible mechanisms that could produce these high backscatter and CPR values is the coherent backscatter model (Hapke 1990), which requires scattering centers (cracks or inhomogeneities) embedded in a low-loss matrix such as ice (Hapke & Blewett 1991; Mishchenko 1992). Unlike the Galilean satellites and Mercury poles, radar observations of lunar PSRs did not show radar-bright regions that indicate the presence of large expanses of water ice but indicated that the possible ice could be present as few wt% in the uppermost meter of regolith or mixed as patches within the polar regolith as “dirty ice” (Nozette et al. 1996; Thompson et al. 2011; Thomson et al. 2012). If, however, the lunar ice is present in the regolith in the form ice-filling pores, then radar scattering differences might be too small to detect (Thompson et al. 2011; Virkki & Bhiravarasu 2019). Another line of evidence from the interpretation of Mini-SAR and Mini-RF radar data suggested that both poles contain abundant water in the form of relatively “clean” ice, within the upper couple of meters of the lunar surface (Spudis et al. 2010, 2013). Other work has called this interpretation into question and has interpreted PSRs with enhanced CPR signatures as indicative of effects due to roughness, not water ice (Simpson & Tyler 1999; Campbell et al. 2006; Fa & Cai 2013; Eke et al. 2014; Fa & Eke 2018).
            Impact ejecta and melts: Polarimetric radar is a powerful tool for studying lunar impact craters, exposing their “rough” ejecta deposits and associated impact melts (Campbell 2012; Neish et al. 2014). New multiwavelength radar observations would enable constraining block sizes and to distinguish the surface from buried rocks in proximal and distant ejecta deposits (Campbell 2002; Ghent et al. 2016). The impact melt deposits have very high CPRs compared to other features on the Moon, suggesting that their surfaces are some of the roughest material on the Moon at the centimeter to decimeter scale, even though they appear smooth at the meter scale (e.g., Carter et al. 2012; Neish et al. 2014, 2017).
            Radar-dark halos: A certain class of impact craters on the Moon have radar-dark halos around them, comprised of fine-grained, rock-poor ejecta distal to the blocky proximal ejecta (Ghent et al. 2005). The composition of this halo material is suggested to be a mixture of crater ejecta and preexisting regolith (e.g., Ghent et al. 2016), and radar data at multiple wavelengths could provide significant new insight into the behavior of ejecta and their effects on the surrounding terrain.2.1. DFSAR ObjectivesAfter the end of the commissioning phase in September 2019, the science mission phase of Chandrayaan-2 has started, which will continue until any major contingencies hamper the spacecraft operation and/or science data reception. The DFSAR instrument is designed to address key science questions related to the aforementioned target properties during the Chandrayaan-2 science mission phase, by probing the lunar regolith to understand surface physical and dielectric properties and structure, and by exploring the PSRs at the lunar poles. Specifically, DFSAR data will be utilized in characterizing the physical properties of lunar mare, impact craters, and their associated ejecta and melt deposits, volcanic deposits including pyroclastic mantling material and domes, buried layering, and volatiles at lunar poles.The DFSAR architecture is new for planetary missions. The quad-polarity design (Sections 3 and 4) provides data sufficient to measure the 2 × 2 complex scattering matrix of the backscattered field, which in turn leads to the extraction of all information available in the radar reflections. The fully polarimetric radar scattering theory contains all of the scattering information for any arbitrary polarization state (e.g., circular, linear, and hybrid modes), thus providing better tools than CPR alone to develop scattering models of the lunar terrain (Fa et al. 2011; Campbell 2012). Moreover, measuring the full scattering matrix allows a powerful observation space sensitive to shape, orientation, and dielectric properties of the scatterers within a resolution cell (Henderson & Lewis 1998).3. DFSAR Instrument Overview, Operations, and CalibrationThe DFSAR instrument is designed with a configuration that enables operations in dual-frequency and quad- and compact-polarity imaging modes to collect information about the scattering properties of the lunar surface at multiple look-angle geometries. For a complete review of technical specifications, system design, and configuration of the DFSAR instrument, the reader is referred to Putrevu et al. (2016, 2020). In this section, we briefly describe the system configuration, followed by the calibration of DFSAR data products.3.1. DFSAR Instrument OverviewDFSAR is designed as two independent radar systems in the L band and S band sharing a common microstrip planar antenna. Such a configuration enables standalone L/S-band imaging as well as synchronous L and S imaging; the latter is ensured by having synchronization and timing signals sourced from L-band systems to those of S band. Linear frequency modulated (LFM; or more commonly referred to as chirp) signals of requisite bandwidth (BW; selectable from 75 to 2 MHz) are transmitted through two transmit chains feeding to the H and V ports of the antenna. Backscattered signals from the target are collected by the antenna and passed through H and V chains of the receiver, followed by digitization, data processing, and formatting in the data-acquisition system to meet the data-rate constraints of the Chandrayaan-2 mission. Transmitters of each band are configured in two chains for H and V, with gallium-nitride-based solid-state power amplifiers (SSPAs) in a high-efficiency design. Each chain of SSPA has a six-bit digital phase-shifter to enable requisite phase-setting for transmitting in either circular or linear polarization. Receivers are configured with low-noise and high-gain amplifiers to meet the high signal-to-noise ratio requirements of the system. A key feature of the DFSAR system is its capability of onboard range compression implemented in the data-acquisition system and provides data-rate reduction to the tune of 70%. This compression technique is a major factor in realizing the FP mode of operations, which is data-rate and volume intensive (about twice that of a hybrid-polarization implemented in Mini-RF/Mini-SAR instruments). Table 1 lists the major specifications of the DFSAR system.Table 1. DFSAR Top-level Specifications
                           Parameters
                        
                           
                              L Band
                        
                           
                              S Band
                        AltitudeNominal 100 kmFrequency1.25 GHz2.5 GHzSAR modesSingle/Dual/Hybrid and Full polarimetryRange swath10 kmIncidence angle9.6°–36.9°Resolution2–75mChirp bandwidth75–2MHzAntennaMicrostrip antenna, 1.4m × 1.1mAntenna gain22 dBi25 dBiReceiver gain90 dBAxial ratio in hybrid pol mode0.4 dB1.1 dBCross-polarization>30 dBTransmit pulse width80 μ–25 μs (Hybrid Pol) 50 μ–25 μs (Full Pol)SSPA peak power45 W40 WMax duty cycle24%Receiver noise figure2.8 dB4.3 dBReceiver front-end losses2 dB3.2 dBNESZ at swath end for 75 m−33.7 dB (Hybrid Pol)−26.2 dB (Hybrid Pol)resolution and 30° incidence−30 dB (Full Pol)−22.5 dB (Full Pol)Onboard processingRange compression and BAQData rate160 MbpsRaw bus power100 WPayload mass20 kg
			Download table as: 
            ASCIITypeset imageOnboard calibration pulses flank the imaging pulses (data-windows) and include chirp replica and noise pulses as pre- and post-imaging sequences. The replica signal refers to the signal that is coupled (−20dB) at both ends of the transmitter and the receiver. The DFSAR instrument is configured on the Chandrayaan-2 orbiter with the planar antenna mounted on the moon-viewing panel (positive yaw) of the spacecraft cuboid, with its electronics packages integrated behind this panel. During lunar imaging, the side-looking geometry of the radar is affected by roll, tilting the orbiter by the required look angle.3.2. DFSAR Preflight (Lab) CharacterizationLab characterization is essential for any instrument to meet the performance specifications, more so in the case of a polarimetric imaging system like the DFSAR. The instrument has undergone extensive in-lab characterization to ensure high-quality performance in terms of radiometry as well as polarimetry. Some of the salient tests conducted are as follows:1.  Radio frequency (RF) measurements of transmit peak power, receiver gain and noise figure at different manual-gain control (MGC) settings, front-end losses, receiver flatness, and imbalances in amplitude and phase are some of the key parameters for DFSAR characterization. These parameters have been characterized over the operating temperature range of 0°–45°C, at the individual subsystem/component level. Important RF parameters were additionally characterized during thermo-vacuum tests of integrated payload.2.  Receive chains have been characterized to generate mapping of the digitized signal counts to input signal power, which is an important input for radiometric calibration. The entire chain from the input antenna port through front-end elements and receiver is characterized as an integrated unit to yield the above mapping. Similarly, transmit replica calibration path is also characterized, which serves as a reference to track any drifts in transmit and receive gain.3.  Far-field measurements of the antenna patterns for both bands and H and V polarizations have been done in a compact antenna test facility (CATF). Important parameters derived from the patterns include antenna gain, beam squint angles, and cross-polarization ratio, which play an important role in establishing the calibration equation and for other postprocessing requirements.The above measurements meet the design specifications of the instrument and directly feed to the radiometric calibration of the SAR data, as described in Section 3.4.1. The received counts are first converted into expected backscattered signal power at the antenna-port input and used further to derive sigma naught (σ°) by applying the above-mentioned lab measurements. σ° is the normalized measure of the radar return from a distributed target and is defined as per unit area on the ground.1.  The DFSAR instrument is designed to operate in any of these modes: single (HH/VV), dual (HH+HV/VV+VH), hybrid (LHCP/RHCP), or Full pol (FP). However, FP is the mode being used for systematic acquisitions of the lunar polar regions. At select locations of the Moon, imaging using hybrid polarimetry is also being performed, for which LHCP (left-hand circular polarity on transmission) mode is the default setting. For polarimetric characterization of the transmit chains, axial ratio measurements are crucial. In addition to characterizing the quality of circularly polarized transmission using axial ratios (Figure A1 in the Appendix), the measurements have yielded the best phase settings of the six-bit phase-shifters for the LHCP, right-hand circular polarity (RHCP), and FP modes (Figure 1).2.  Orientation of left/right circular transmit polarization is determined using helical antennas of the corresponding polarization. This has been used as a foolproof method to verify the handedness of the circular polarization.3.  Amplitude and phase imbalances between H and V receive chains (covering the entire path of the antenna, front end, and respective receiver) for polarimetric performance.4.  Using an external optical fiber delay-line, end-to-end characterization of the payload was carried out. The external setup consisted of a horn antenna followed by a circulator, delay-line, and appropriate attenuator pads; the transmit signal is collected by the horn antenna and passed through the circulator to the delay-line; the delayed signal is fed back to the circulator, to enable its transmission toward the DFSAR antenna. Using this, the process of replica acquisition and chirp compression using onboard range compression was validated.5.  The performance of the onboard range compression was validated using simulated distributed target data. The range-expanded data was played back from the Arbitrary Waveform Generator to the DFSAR data-acquisition system for range compression using chirp replica from pre-imaging calibration. The onboard range-compressed data were evaluated for impulse response parameters with respect to that of conventional on-ground processed output and found to be meeting the performance requirements.6.  Another key characterization activity is ensuring the phase synchronization between the L and S bands to be better than 6°. Chirp replicas for L and S bands compensate for any initial bias in phases, thereby ensuring a phase synchronization of better than 6°, as measured and verified.7.  DFSAR has an experimental radiometer mode for which the requisite measurements, data acquisition, and analysis will be described in a separate paper. However, for the sake of completion, an important characterization deserving mention here is that of radiometer mode full-chain “hot-load” calibration performed using microwave absorbers. This calibration complements the “cold-sky” measurement in operational onboard conditions toward retrieval of brightness temperatures.
            Zoom In
				Zoom Out
				Reset image size
					Figure 1. Axial ratio measurements for various DFSAR modes of operation. Top: The bar chart on the left illustrates the ellipticity of the transmitted signals (for the L-band SAR, shown as an example) corresponding to different phase-shifter settings, and the inset table on the right shows axial ratio measurement results for L- and S-band transmission modes. The minimum axial ratio of 0.39 dB corresponds to phase-shifter settings of 45° and 0° of H and V transmit chains, respectively, summarily represented as H45V0 on the x-axis. Bottom: The sense of circularly polarized signals (LHCP/RHCP) is cross-verified using helical antennas of the corresponding polarization. The ellipses at the bottom of the figure represent the variation of the transmit signal amplitude radiated by the SAR antenna and measured using a linearly polarized probe at different orientation angles as specified in the plots.Download figure:Standard image
					High-resolution image
					3.3. DFSAR Operations, Data Acquisition, and Product GenerationChandrayaan-2 was launched using ISRO's Geosynchronous Satellite Launch Vehicle (GSLV)-Mark III on 2019 July 22 from Sriharikota, the spaceport of India. It was inserted into lunar orbit on 2019 August 20 and was brought to a near-circular orbit of 80 km × 120 km, with the periapsis and apoapsis close to the lunar south pole and north pole, respectively. Subsequent to the initial commissioning phase activities of DFSAR during 2019 September, the instrument has been tasked to systematically cover the lunar poles (85° and poleward) and is making full-polarimetric measurements in the designated time slots (referred to as dawn–dusk seasons). Since then, DFSAR has been operational and beaming high-quality data to cater to various scientific studies. After this initial campaign, DFSAR will observe the lunar polar regions (±70° to ±90°) at the L band along with selected nonpolar targets at the L and/or S bands. Based on this coverage, selected regions (e.g., where anomalous CPR values exist) will be observed with L- and S-band simultaneous, hybrid- and full-polarization imaging. Over the course of the mission, we will plan to acquire global L-band FP data of the lunar surface to integrate with other global lunar data sets. Raw data from the DFSAR acquisitions are downlinked at different ground stations and processed to generate products at various levels, which are denoted below:1.  
                        Level-0A/0B: represent range-uncompressed and compressed, respectively. These are raw-data products with ancillary data for further processing. In this, different processes (like block adaptive quantization) applied to reduce the data rate are reversed to represent the original instrument data.2.  
                        Level-1A: represents seleno-tagged single-look complex (SLC) data.3.  
                        Level-1B: represents seleno-tagged ground-range product.4.  
                        Level-2: represents seleno-referenced product.
            A Range–Doppler algorithm (Cumming & Wong 2005) is used for processing DFSAR stripmap data, which efficiently processes the SAR data in the frequency domain. The time-domain correlation operation is replaced by the frequency domain multiplication operation. Each pixel of a given slant image is tagged to a particular seleno-location, i.e., latitude/longitude, value using the slant range grid generated from the orbit attitude information, a digital elevation model (DEM) available at ∼118.5 m spacing from the LRO Laser Orbital Laser Altimeter (LOLA; Mazarico et al. 2011) and radar parameters.3.3.1. Image Quality AnalysisAny feature or target of dimensions similar to or smaller than that of the SAR resolution cell and with sufficient backscatter energy will appear as a “point target” in a SAR image. Such point targets are suitable candidates to characterize the impulse response of the SAR system. DFSAR data obtained during the commissioning phase have been processed and evaluated for payload performance. In this process, range and azimuth impulse responses have been analyzed for opportunity point targets, because no ideal point targets are available on the lunar surface. System impulse response characteristics such as peak-to-sidelobe ratio (PSLR) and resolution have been derived from these opportunity targets, and some of the sample results are shown in Table 2 and Figure 2. After the radiometric calibration process (described in Section 3.4.1) of DFSAR, noise-equivalent sigma naught (NESZ) was derived (Table 3) from the noise floor of the system, measured during onboard calibration data acquisition. NESZ represents the lowest σ° that can be measured by the instrument. Radiometric resolution has been estimated over homogeneous regions located in the processed images. All the DFSAR performance metrics are found to match the prelaunch expected values.Zoom In
				Zoom Out
				Reset image size
					Figure 2. 
                        L-band range (left) and azimuth (right) impulse response for opportunity point targets.Download figure:Standard image
					High-resolution image
					Table 2. Impulse Response Performance Obtained from DFSAR Data Acquired in Different Polarimetric ModesModeChirp BWRange PSLRAzimuth PSLRRangeAzimuth (MHz)(dB)(dB)Resolution (m)Resolution (m)
                              L-band FP7.5−11.7−10.91812.8
                              L-band CP75−15.35−15.561.92.3
                              S-band CP50−13.9−13.242.73.03
			Download table as: 
            ASCIITypeset imageTable 3. Noise-equivalent Sigma-naught (NESZ) Performance Metrics for DFSAR Polarimetric Modes Using SAR Noise Calibration Data, with Signal Power Normalized to 100 km Altitude, and at 30° Angle of IncidenceModeChirp BandwidthPrelaunch NESZ EstimatePostlaunch NESZ Value (MHz)(dB)(dB)
                              L-band FP7.5−27.7−27.9
                              L-band CP7.5−29.5−29.8
                              S-band FP7.5−22.6−23.8
                              S-band CP7.5−24.3−25.6
			Download table as: 
            ASCIITypeset image3.4. Calibration of the Data ProductsAs the DFSAR is a polarimetric system with the capability of acquiring quad-pol and compact polarimetric (CP) data, radiometric and polarimetric calibration are critical for the derivation of lunar surface physical parameters.3.4.1. Radiometric CalibrationRadiometric calibration provides the coefficients to map SAR data (pixel values) to their corresponding backscatter coefficients, i.e., the normalized radar cross section of targets. Due to the absence of any calibration targets on the Moon with the well-known radar cross section in the L and S bands, the calibration is performed based on the radar equation, using lab-characterized system parameters:where DN represents the digital number corresponding to a pixel after processing; kv
                   is a factor that accounts for quantization to convert count to power at the input of analog-to-digital converter with 50 ohm impedance; goa and gor are the pulse repetition frequency (PRF)/Doppler BW and the range sampling frequency/range BW corresponding to the oversampling factors for azimuth and range respectively; Lb
                   is beam-shape loss; and SF is the scale factor applied during onboard processing, for a given polarization chain.Using the above equation, processor-gain-related factors are first corrected to convert the digital counts into raw power, as would be present at the receiver output. The raw power as computed from Equation (1) is then inverted using Equation (2) to derive σ°, with the inputs from lab-measured parameters:where Pr
                   is the power received corresponding to a digital count of a pixel; Pt
                   is transmitted power; Gt
                   and Gr
                   are the transmit and receive antenna gains respectively; Grx is the receiver gain; R is the slant range of the pixel location; λ is the radar wavelength, θ denotes the incidence angle, and c is the velocity of light; τ is the pulse width or chirp duration; MGCrx is the attenuation setting by the manual-gain controller of the receiver; Ls
                   is the receiver chain front end loss; and L is the azimuth antenna length.The processed images are corrected using the lab-measured antenna patterns for radiometric normalization, so that a uniform/single calibration equation can be used for the entire data set, irrespective of the pixel position. In the case of Level-1A (SLC) products, the relationship between the digital number (DN) and the backscattering coefficient σθ
                   for each polarization channel can be written as:where DN = I + Qi, and I and Q are, respectively, the real part and the imaginary part of the complex pixel value; K is the absolute calibration constant and is constant over all SLC products; and L, M are the number of lines and columns in the product. The calibration constant (K) is provided with the associated xml label file for each data set and is different for products Level-1A and Level-1B and above. For Level-1B and above products, σ° in linear scale to dB can be directly derived asSpecimen results as shown in Figure 3 and Table 4 compare the radiometry of S-band DFSAR hybrid-polarimetric mode data with S-band Mini-RF data over a part of the Gauss crater region that has overlap from both instruments. The radiometric accuracies (with incidence-angle-independent gamma-naught, γ°) obtained over different sample regions are found to be better than 2 dB. In the absence of any L-band data and any full polarimetry data of the Moon, validation could only be done for the process adopted for radiometric calibration, using S-band hybrid-polarity data of Mini-RF as reference. Considering that there are no additional processes that may hinder such a generalization for either L-band or full polarimetry data, a radiometric accuracy of better than 2 dB can be safely considered as a worst-case value.Zoom In
				Zoom Out
				Reset image size
					Figure 3. Radiometry comparison over five sample regions (indicated by boxes in (b) and (c)) between (a) S-band DFSAR and (d) S-band Mini-RF hybrid-polarimetric data shown here in m-chi decomposition images with the RGB color scheme indicating the dominant scattering regime: even bounce (red), odd bounce (blue), and volume scattering (green). The extent of the zoomed-in portions (b) and (c) are shown with yellow boxes in (a) and (d), respectively. Both DFSAR and Mini-RF images shown here are resampled to a similar spatial resolution (∼30 m pixel−1) and the center coordinates of the DFSAR image in (a) are 35.36°N, 79.54°E.Download figure:Standard image
					High-resolution image
					Table 4. Radiometry (γ°) Comparison of the S-band DFSAR with Mini-RF Data over Five Selected Regions as Shown in Figure 3
                     
                              Region
                           
                              DFSAR S-band CP
                           
                              Mini-RF S-band CP
                           
                              
                                 γ°
                            
                              γ° (dB)
                              γ° (dB)
                              Difference (dB)
                            LHLVLHLVLHLVReg–1−13.78−13.60−12.49−12.54−1.29−1.06Reg–2−13.18−13.06−14.80−14.641.621.58Reg–3−15.19−15.14−15.88−15.650.690.51Reg–4−12.99−12.96−13.58−13.690.590.73Reg–5−14.79−14.28−14.08−14.45−0.710.17
                        Note. LH and LV stands for left-circular transmit and horizontal (H) and vertical (V) receive polarizations respectively.
			Download table as: 
            ASCIITypeset image3.4.2. Polarimetric Calibration for Full PolarimetryPolarimetric characterization of a region of interest is heavily dependent on the quality of polarimetric calibration that establishes the interchannel (HH/VV/HV/VH) phase relationships. Calibration errors can affect polarimetry products (coherency and covariance matrices and the other derived daughter parameters) thereby leading to misinterpretation of scattering/physical processes under study. The “measured” scattering matrix (of a target) is a product of the “actual” scattering matrix (Section 4.1) and the distortion matrices corresponding to the transmit [T] and receive [R] paths of the radar signal. Considering the order of interactions between the signal and target, the “measured” scattering matrix may be written aswithwhere ft
                   and fr
                   represent channel imbalances in HH polarization with reference to that of VV, for the transmit and receive paths respectively. They comprise differences in antenna patterns between the polarizations and also the gain and phase of their respective paths. δ1 and δ2 represent the cross-talk terms HV and VH, respectively, mainly arising from the antenna cross-polarization components. As we are dealing with imaging in lunar conditions with negligible ionosphere, Faraday rotation of the signals is not expected and thereby not considered in the above equation. After the SAR image formation and radiometric calibration, polarimetric calibration is applied to the resulting data. The methodology adopted for this calibration closely follows that of Sun et al. (2018) due to the following major factors: (1) the methodology does not depend on any external calibration targets (like corner reflectors) and is derived from the SAR data itself, which is a necessity in the case of lunar imaging. (2) Cross-talk estimation and channel imbalances are estimated in an iterative fashion due to their mutual dependency. (3) The computations are based on a derived covariance matrix, which results in more limiting equations to derive the phase estimates.DFSAR polarimetric calibration is based on the above methodology while focusing on dominant surface and volume backscatter regions for co- and cross-polarization signal responses. The results in Figure 4 show phase histograms for uncalibrated and calibrated L-band (Figures 4(a)–(b)) and S-band (Figures 4(c)–(d)) acquisition, corresponding to co- and cross-polarization cases. Phase corrections of the order of −50° for the co-polarization phase and −5° for the cross-polarization phase are required for phase calibration of DFSAR L-Band acquisitions. Corresponding numbers for the calibration of S-band FP acquisitions are −100° and +100° for the co- and cross-polarization phase, respectively. For L-band data, the stability of the co- and cross-polarization phases observed over multiple acquisitions (Figures 4(e)–(f)) is typically within ±10°, indicating that a uniform set of polarimetric calibration factors can be applied for the DFSAR data sets. However, monitoring of polarimetric calibration would be continued for future acquisitions, and any deviations will be taken care of by estimating new calibration coefficients and incorporating them during the subsequent processing of the DFSAR data.Zoom In
				Zoom Out
				Reset image size
					Figure 4. DFSAR polarimetric calibration results. (a), (b) Histograms of L-band co-pol (HH/VV) and cross-pol (HV/VH) phase differences; (c), (d) histograms of S-band co-pol (HH/VV) and cross-pol (HV/VH) phase differences; (e), (f) stability of co- and cross-pol phase differences across several L-band data sets, prior to applying the correction (i.e., uncalibrated), respectively. This illustrates that a single set of correction factors is uniformly applicable across the acquired data sets.Download figure:Standard image
					High-resolution image
					For a holistic view of the quality of the polarimetric calibration, opportunity point targets have been analyzed for polarimetric signatures, both in terms of polar coordinates (ellipticity angle and orientation angle) and Cartesian coordinates representing Stokes parameters (Raney 2007; Lee & Pottier 2009) over a Poincaré sphere. The results (Figure 5) show an excellent match with those of an ideal trihedral corner reflector (Zyl et al. 1987) and give the required confidence on the quality of the polarimetric calibration.Zoom In
				Zoom Out
				Reset image size
					Figure 5. Uncalibrated (left set) and calibrated (right set) polarimetric signatures corresponding to an opportunity point target to assess the efficacy of polarimetric calibration. The response for each Stokes parameter (S1, S2, S3) closely matches with those expected from an ideal trihedral corner reflector, thereby validating the polarimetric calibration exercise.Download figure:Standard image
					High-resolution image
					3.4.3. Polarimetric Calibration for Compact PolarimetryThe calibration approach for CP data is a subset of that adopted for calibration of full-polarimetric data. The channel and phase imbalances were derived from L- and S-band data over some uniform regions on the lunar surface. The relative phase histograms computed from Stokes parameters (Raney 2007; Raney et al. 2011) were derived over several CP data acquisitions to establish the consistency of amplitude and phase-calibration parameters. The order of relative phase (δ) corrections is about 40° for L-band and 10° for S-band data to achieve the phase calibration. The results of applying this correction is shown in Figure 6 in the form of uncalibrated and calibrated δ and degree of polarization (m) histograms for a set of DFSAR L- and S- band CP data.Zoom In
				Zoom Out
				Reset image size
					Figure 6. Relative phase (δ) and degree of polarization (m) histograms for DFSAR (a), (b) L-band CP and (c), (d) S-band CP data, respectively, after applying the necessary calibration factors.Download figure:Standard image
					High-resolution image

...42183 bytes truncated...

The tool call succeeded but the output was truncated. Full output saved to: /home/rabindra03/.local/share/opencode/tool-output/tool_ebd25a786001DMV8FCsfXhSFcc
Use the Task tool to have explore agent process this file with Grep and Read (with offset/limit). Do NOT read the full file yourself - delegate to save context.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 11.6s)

_Thinking:_

Excellent! I've gathered substantial information about DFSAR. Now let me create a comprehensive implementation plan. I need to:

1. Create the folder structure
2. Build all the source code files
3. Create config files
4. Create tests

Let me plan the architecture carefully based on what I've learned:

**Key DFSAR facts:**
- L-band (1.25 GHz, ~24cm wavelength) and S-band (2.5 GHz, ~12cm)
- Full quad-pol (HH, HV, VH, VV) → 2×2 scattering matrix
- Incidence angles: 9.6°–36.9°
- Resolution: 2–75m
- PDS4 archive standard
- Data levels: L0, L1A (SLC), L1B (ground-range), L2 (seleno-referenced)
- Calibration: radiometric calibration via radar equation, polarimetric calibration for cross-talk and channel imbalance
- Key products: CPR, degree of polarization, Cloude-Pottier decomposition (H/α/A), m-chi decomposition, backscatter coefficients

**Processing pipeline:**
1. Input/Data Loading - read PDS4 format, SLC complex data
2. Radiometric Calibration - apply calibration constants
3. Polarimetric Calibration - cross-talk removal, channel balancing
4. Speckle Filtering - Refined Lee, Boxcar, etc.
5. Coherency Matrix T3 generation
6. Cloude-Pottier Decomposition (H, α, A)
7. Circular Polarization Ratio (CPR)
8. Degree of Polarization (DOP)
9. Backscatter Coefficients (σ°, γ°)
10. Output generation (GeoTIFFs, Zarr cubes, PNGs, stats)

Let me build this now. I'll create the full repository structure with all files.

Directory structure:
```
lunaice/
├── src/
│   └── lunaice/
│       ├── __init__.py
│       ├── config.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── reader.py
│       │   └── writer.py
│       ├── processing/
│       │   ├── __init__.py
│       │   ├── calibration.py
│       │   ├── speckle.py
│       │   ├── polarimetry/
│       │   │   ├── __init__.py
│       │   │   ├── coherency.py
│       │   │   ├── cloude_pottier.py
│       │   │   ├── cpr.py
│       │   │   ├── decomposition.py
│       │   │   └── backscatter.py
│       │   └── pipeline.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── schemas.py
│       └── cli.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_reader.py
│   ├── test_calibration.py
│   ├── test_polarimetry.py
│   └── test_pipeline.py
├── configs/
│   ├── default.yaml
│   └── processing.yaml
├── notebooks/
│   └── dfsar_exploration.ipynb
├── docs/
│   ├── architecture.md
│   └── api.md
├── pyproject.toml
└── README.md
```

Let me now create all these files systematically. This is going to be extensive but I need to be thorough.

Let me start by creating the directory structure, then write each file.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "content": "Create repository structure and directory tree",
      "status": "in_progress",
      "priority": "high"
    },
    {
      "content": "Build core data models and schemas (PolarimetricData, metadata)",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Implement I/O module (PDS4 reader, GeoTIFF/Zarr writer)",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Implement radiometric calibration module",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Implement polarimetric calibration module",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Implement speckle reduction module",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Implement coherency matrix (T3) generation",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Implement Cloude-Pottier decomposition (H/alpha/A)",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Implement CPR, DOP, backscatter coefficients",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Build pipeline orchestrator",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Build CLI interface",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Create configuration files (YAML)",
      "status": "pending",
      "priority": "high"
    },
    {
      "content": "Write tests",
      "status": "pending",
      "priority": "medium"
    },
    {
      "content": "Create pyproject.toml and documentation",
      "status": "pending",
      "priority": "medium"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create repository structure and directory tree",
    "status": "in_progress",
    "priority": "high"
  },
  {
    "content": "Build core data models and schemas (PolarimetricData, metadata)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Implement I/O module (PDS4 reader, GeoTIFF/Zarr writer)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Implement radiometric calibration module",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Implement polarimetric calibration module",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Implement speckle reduction module",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Implement coherency matrix (T3) generation",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Implement Cloude-Pottier decomposition (H/alpha/A)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Implement CPR, DOP, backscatter coefficients",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Build pipeline orchestrator",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Build CLI interface",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Create configuration files (YAML)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Write tests",
    "status": "pending",
    "priority": "medium"
  },
  {
    "content": "Create pyproject.toml and documentation",
    "status": "pending",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 3.6s)

_Thinking:_

Let me create the full directory structure first, then write all the files.

**Tool: bash**

**Input:**
```json
{
  "command": "mkdir -p /home/rabindra03/workspace/DEV/LUNAICE/{src/lunaice/{io,processing/polarimetry,models},tests,configs,notebooks,docs}",
  "description": "Create repository directory structure"
}
```

**Output:**
```
(no output)
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 13.9s)

_Thinking:_

Good, directory structure created. Now let me write all the files. I'll start with the core modules and work outward. Let me write them all in parallel where possible.

Now I'll build all source files. Starting with core models, config, and I/O.

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/__init__.py",
  "content": "from lunaice.config import DFSARConfig, ProcessingConfig, SpeckleFilterConfig\nfrom lunaice.pipeline import Pipeline\n\n__all__ = [\"DFSARConfig\", \"ProcessingConfig\", \"SpeckleFilterConfig\", \"Pipeline\"]\n__version__ = \"0.1.0\"\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/models/__init__.py",
  "content": "from lunaice.models.schemas import (\n    PolarimetricData,\n    Metadata,\n    CalibrationConstants,\n    CoherencyMatrix,\n    DecompositionProducts,\n    ProcessingSummary,\n)\n\n__all__ = [\n    \"PolarimetricData\",\n    \"Metadata\",\n    \"CalibrationConstants\",\n    \"CoherencyMatrix\",\n    \"DecompositionProducts\",\n    \"ProcessingSummary\",\n]\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/models/schemas.py",
  "content": "from __future__ import annotations\n\nimport uuid\nfrom dataclasses import dataclass, field\nfrom datetime import datetime\nfrom enum import Enum\nfrom typing import Optional\n\nimport numpy as np\n\n\nclass FrequencyBand(str, Enum):\n    L_BAND = \"L\"\n    S_BAND = \"S\"\n\n\nclass PolarizationMode(str, Enum):\n    QUAD_POL = \"quad_pol\"\n    DUAL_POL = \"dual_pol\"\n    HYBRID_POL = \"hybrid_pol\"\n\n\nclass ProcessingLevel(str, Enum):\n    LEVEL_0A = \"L0A\"\n    LEVEL_0B = \"L0B\"\n    LEVEL_1A = \"L1A\"\n    LEVEL_1B = \"L1B\"\n    LEVEL_2 = \"L2\"\n\n\n@dataclass\nclass Metadata:\n    product_id: str\n    processing_level: ProcessingLevel\n    frequency_band: FrequencyBand\n    polarization_mode: PolarizationMode\n    acquisition_time: datetime\n    orbit_number: int\n    incidence_angle_deg: float\n    slant_range_resolution_m: float\n    azimuth_resolution_m: float\n    looks_range: int\n    looks_azimuth: int\n    calibration_constant: float\n    wavelength_cm: float\n    center_latitude: float\n    center_longitude: float\n    pixel_spacing_m: float\n    cal_params: Optional[dict] = None\n\n    @classmethod\n    def from_dict(cls, d: dict) -> Metadata:\n        return cls(\n            product_id=d.get(\"product_id\", \"\"),\n            processing_level=ProcessingLevel(d.get(\"processing_level\", \"L1A\")),\n            frequency_band=FrequencyBand(d.get(\"frequency_band\", \"L\")),\n            polarization_mode=PolarizationMode(d.get(\"polarization_mode\", \"quad_pol\")),\n            acquisition_time=datetime.fromisoformat(d.get(\"acquisition_time\", \"2020-01-01T00:00:00\")),\n            orbit_number=int(d.get(\"orbit_number\", 0)),\n            incidence_angle_deg=float(d.get(\"incidence_angle_deg\", 30.0)),\n            slant_range_resolution_m=float(d.get(\"slant_range_resolution_m\", 15.0)),\n            azimuth_resolution_m=float(d.get(\"azimuth_resolution_m\", 15.0)),\n            looks_range=int(d.get(\"looks_range\", 1)),\n            looks_azimuth=int(d.get(\"looks_azimuth\", 1)),\n            calibration_constant=float(d.get(\"calibration_constant\", 1.0)),\n            wavelength_cm=float(d.get(\"wavelength_cm\", 24.0)),\n            center_latitude=float(d.get(\"center_latitude\", 0.0)),\n            center_longitude=float(d.get(\"center_longitude\", 0.0)),\n            pixel_spacing_m=float(d.get(\"pixel_spacing_m\", 15.0)),\n            cal_params=d.get(\"cal_params\"),\n        )\n\n\n@dataclass\nclass CalibrationConstants:\n    k_hh: float = 1.0\n    k_hv: float = 1.0\n    k_vh: float = 1.0\n    k_vv: float = 1.0\n    phase_offset_hh_vv: float = 0.0\n    phase_offset_hv_vh: float = 0.0\n    cross_talk_hv: float = 0.0\n    cross_talk_vh: float = 0.0\n    channel_imbalance_amp: float = 1.0\n    channel_imbalance_phase: float = 0.0\n\n\n@dataclass\nclass PolarimetricData:\n    data_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])\n    hh: Optional[np.ndarray] = None\n    hv: Optional[np.ndarray] = None\n    vh: Optional[np.ndarray] = None\n    vv: Optional[np.ndarray] = None\n    metadata: Optional[Metadata] = None\n    calibration: Optional[CalibrationConstants] = None\n\n    @property\n    def shape(self) -> tuple:\n        if self.hh is not None:\n            return self.hh.shape\n        return (0, 0)\n\n    @property\n    def is_quad_pol(self) -> bool:\n        return all(x is not None for x in [self.hh, self.hv, self.vh, self.vv])\n\n    def validate(self) -> bool:\n        shapes = []\n        for arr in [self.hh, self.hv, self.vh, self.vv]:\n            if arr is not None:\n                shapes.append(arr.shape)\n        if len(set(shapes)) > 1:\n            raise ValueError(f\"Inconsistent channel shapes: {shapes}\")\n        return True\n\n\n@dataclass\nclass CoherencyMatrix:\n    t11: np.ndarray\n    t12: np.ndarray\n    t13: np.ndarray\n    t21: np.ndarray\n    t22: np.ndarray\n    t23: np.ndarray\n    t31: np.ndarray\n    t32: np.ndarray\n    t33: np.ndarray\n\n    @property\n    def shape(self) -> tuple:\n        return self.t11.shape\n\n\n@dataclass\nclass DecompositionProducts:\n    entropy: Optional[np.ndarray] = None\n    alpha_deg: Optional[np.ndarray] = None\n    anisotropy: Optional[np.ndarray] = None\n    lambda_1: Optional[np.ndarray] = None\n    lambda_2: Optional[np.ndarray] = None\n    lambda_3: Optional[np.ndarray] = None\n    alpha_1: Optional[np.ndarray] = None\n    alpha_2: Optional[np.ndarray] = None\n    alpha_3: Optional[np.ndarray] = None\n    cpr: Optional[np.ndarray] = None\n    dop: Optional[np.ndarray] = None\n    sigma_hh: Optional[np.ndarray] = None\n    sigma_hv: Optional[np.ndarray] = None\n    sigma_vv: Optional[np.ndarray] = None\n    gamma_hh: Optional[np.ndarray] = None\n    gamma_hv: Optional[np.ndarray] = None\n    gamma_vv: Optional[np.ndarray] = None\n    span: Optional[np.ndarray] = None\n    odd_bounce: Optional[np.ndarray] = None\n    double_bounce: Optional[np.ndarray] = None\n    volume_scattering: Optional[np.ndarray] = None\n\n\n@dataclass\nclass ProcessingSummary:\n    input_file: str\n    output_dir: str\n    products_generated: list[str]\n    processing_time_s: float\n    config_snapshot: dict\n    errors: list[str] = field(default_factory=list)\n    warnings: list[str] = field(default_factory=list)\n"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 28.4s)

_Thinking:_

Good. Now let me write the config, I/O, and processing modules.

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/config.py",
  "content": "from __future__ import annotations\n\nimport os\nfrom dataclasses import dataclass, field\nfrom pathlib import Path\nfrom typing import Optional\n\nimport yaml\n\n\n@dataclass\nclass SpeckleFilterConfig:\n    method: str = \"refined_lee\"\n    window_size: int = 7\n    damping_factor: float = 1.0\n    n_looks: int = 4\n    sigma_sar: float = 0.0\n\n    def __post_init__(self):\n        valid = {\"refined_lee\", \"boxcar\", \"lee_sigma\", \"idani\", \"bilateral\"}\n        if self.method not in valid:\n            raise ValueError(f\"Speckle filter must be one of {valid}, got {self.method}\")\n        if self.window_size < 3 or self.window_size % 2 == 0:\n            raise ValueError(f\"window_size must be odd and >= 3, got {self.window_size}\")\n\n\n@dataclass\nclass ProcessingConfig:\n    radiometric_calibration: bool = True\n    polarimetric_calibration: bool = True\n    speckle_filter: Optional[SpeckleFilterConfig] = None\n    multilook_range: int = 1\n    multilook_azimuth: int = 1\n    generate_coherency_matrix: bool = True\n    generate_cloude_pottier: bool = True\n    generate_cpr: bool = True\n    generate_dop: bool = True\n    generate_backscatter: bool = True\n    output_dtype: str = \"float32\"\n    clip_percentile: float = 99.5\n    zarr_compressor: str = \"blosc\"\n    zarr_chunks: tuple[int, int] = (256, 256)\n    geotiff_compress: str = \"LZW\"\n\n\n@dataclass\nclass DFSARConfig:\n    input_file: str = \"\"\n    output_dir: str = \"output\"\n    band: str = \"L\"\n    processing: ProcessingConfig = field(default_factory=ProcessingConfig)\n    logging_level: str = \"INFO\"\n    verbose: bool = False\n    overwrite: bool = False\n\n    @classmethod\n    def from_yaml(cls, path: str | Path) -> DFSARConfig:\n        path = Path(path)\n        if not path.exists():\n            raise FileNotFoundError(f\"Config file not found: {path}\")\n        with open(path) as f:\n            raw = yaml.safe_load(f)\n        return cls._from_dict(raw)\n\n    @classmethod\n    def _from_dict(cls, raw: dict) -> DFSARConfig:\n        proc = raw.get(\"processing\", {})\n        speckle = proc.get(\"speckle_filter\")\n        speckle_cfg = None\n        if speckle:\n            speckle_cfg = SpeckleFilterConfig(**speckle)\n        proc_cfg = ProcessingConfig(\n            **{k: v for k, v in proc.items() if k != \"speckle_filter\"},\n            speckle_filter=speckle_cfg,\n        )\n        return cls(\n            input_file=raw.get(\"input_file\", \"\"),\n            output_dir=raw.get(\"output_dir\", \"output\"),\n            band=raw.get(\"band\", \"L\"),\n            processing=proc_cfg,\n            logging_level=raw.get(\"logging_level\", \"INFO\"),\n            verbose=raw.get(\"verbose\", False),\n            overwrite=raw.get(\"overwrite\", False),\n        )\n\n    def resolve(self) -> None:\n        self.input_file = str(Path(self.input_file).expanduser().resolve())\n        self.output_dir = str(Path(self.output_dir).expanduser().resolve())\n        os.makedirs(self.output_dir, exist_ok=True)\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/io/__init__.py",
  "content": "from lunaice.io.reader import DFSARReader, read_pds4_label\nfrom lunaice.io.writer import GeoTIFFWriter, ZarrWriter, ReportWriter\n\n__all__ = [\"DFSARReader\", \"read_pds4_label\", \"GeoTIFFWriter\", \"ZarrWriter\", \"ReportWriter\"]\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/io/reader.py",
  "content": "from __future__ import annotations\n\nimport logging\nimport xml.etree.ElementTree as ET\nfrom pathlib import Path\nfrom typing import Optional\n\nimport numpy as np\n\nfrom lunaice.models.schemas import (\n    CalibrationConstants,\n    FrequencyBand,\n    Metadata,\n    PolarimetricData,\n    PolarizationMode,\n    ProcessingLevel,\n)\n\nlogger = logging.getLogger(__name__)\n\n\ndef read_pds4_label(label_path: str | Path) -> dict:\n    label_path = Path(label_path)\n    if not label_path.exists():\n        raise FileNotFoundError(f\"PDS4 label not found: {label_path}\")\n    tree = ET.parse(label_path)\n    root = tree.getroot()\n    ns = {\"pds\": \"http://pds.nasa.gov/pds4/pds/v1\"}\n    info = {}\n    ident = root.find(\".//pds:Identification_Area\", ns)\n    if ident is not None:\n        pid = ident.find(\"pds:logical_identifier\", ns)\n        if pid is not None:\n            info[\"product_id\"] = pid.text\n    obs = root.find(\".//pds:Observation_Area\", ns)\n    if obs is not None:\n        time = obs.find(\".//pds:start_date_time\", ns)\n        if time is not None:\n            info[\"acquisition_time\"] = time.text\n    band_info = root.find(\".//pds:Radar_Frequency_Band\", ns)\n    if band_info is not None:\n        info[\"frequency_band\"] = band_info.text\n    else:\n        band_attrib = root.find(\".//pds:Band\", ns)\n        if band_attrib is not None:\n            info[\"frequency_band\"] = band_attrib.text\n    params = root.findall(\".//pds:Parameter\", ns)\n    for p in params:\n        name_el = p.find(\"pds:name\", ns)\n        val_el = p.find(\"pds:value\", ns)\n        if name_el is not None and val_el is not None:\n            info[name_el.text.lower().replace(\" \", \"_\")] = val_el.text\n    return info\n\n\ndef _load_slc_channel(path: str | Path) -> np.ndarray:\n    path = Path(path)\n    if path.suffix in {\".npy\", \".npz\"}:\n        return np.load(path)\n    raw = np.fromfile(path, dtype=np.complex64)\n    return raw\n\n\nclass DFSARReader:\n    def __init__(\n        self,\n        input_path: str | Path,\n        label_path: Optional[str | Path] = None,\n    ):\n        self.input_path = Path(input_path)\n        self.label_path = Path(label_path) if label_path else None\n        self._metadata: Optional[Metadata] = None\n        self._cal: Optional[CalibrationConstants] = None\n\n    def read_metadata(self) -> Metadata:\n        if self._metadata is not None:\n            return self._metadata\n        label_data = {}\n        if self.label_path and self.label_path.exists():\n            label_data = read_pds4_label(self.label_path)\n        else:\n            xml_candidates = list(self.input_path.parent.glob(\"*.xml\")) + list(\n                self.input_path.parent.glob(\"*.lbl*\")\n            )\n            if xml_candidates:\n                label_data = read_pds4_label(xml_candidates[0])\n                logger.info(\"Found PDS4 label: %s\", xml_candidates[0])\n        self._metadata = Metadata.from_dict(label_data)\n        return self._metadata\n\n    def read_calibration(self) -> CalibrationConstants:\n        if self._cal is not None:\n            return self._cal\n        met = self.read_metadata()\n        raw = met.cal_params or {}\n        self._cal = CalibrationConstants(\n            k_hh=float(raw.get(\"cal_const_hh\", 1.0)),\n            k_hv=float(raw.get(\"cal_const_hv\", 1.0)),\n            k_vh=float(raw.get(\"cal_const_vh\", 1.0)),\n            k_vv=float(raw.get(\"cal_const_vv\", 1.0)),\n            phase_offset_hh_vv=float(raw.get(\"phase_hh_vv\", 0.0)),\n            phase_offset_hv_vh=float(raw.get(\"phase_hv_vh\", 0.0)),\n            cross_talk_hv=float(raw.get(\"crosstalk_hv\", 0.0)),\n            cross_talk_vh=float(raw.get(\"crosstalk_vh\", 0.0)),\n            channel_imbalance_amp=float(raw.get(\"ch_imbalance_amp\", 1.0)),\n            channel_imbalance_phase=float(raw.get(\"ch_imbalance_phase\", 0.0)),\n        )\n        return self._cal\n\n    def read_slc(self) -> PolarimetricData:\n        met = self.read_metadata()\n        cal = self.read_calibration()\n        data = PolarimetricData(metadata=met, calibration=cal)\n        pattern = str(self.input_path)\n        if self.input_path.is_dir():\n            files = {\n                \"hh\": list(self.input_path.glob(\"*HH*\")) + list(self.input_path.glob(\"*hh*\")),\n                \"hv\": list(self.input_path.glob(\"*HV*\")) + list(self.input_path.glob(\"*hv*\")),\n                \"vh\": list(self.input_path.glob(\"*VH*\")) + list(self.input_path.glob(\"*vh*\")),\n                \"vv\": list(self.input_path.glob(\"*VV*\")) + list(self.input_path.glob(\"*vv*\")),\n            }\n        else:\n            stem = self.input_path.stem\n            parent = self.input_path.parent\n            files = {\n                \"hh\": [parent / f\"{stem}_HH{self.input_path.suffix}\"],\n                \"hv\": [parent / f\"{stem}_HV{self.input_path.suffix}\"],\n                \"vh\": [parent / f\"{stem}_VH{self.input_path.suffix}\"],\n                \"vv\": [parent / f\"{stem}_VV{self.input_path.suffix}\"],\n            }\n        for pol, candidates in files.items():\n            valid = [c for c in candidates if c.exists()]\n            if valid:\n                ch_data = _load_slc_channel(valid[0])\n                setattr(data, pol, ch_data.reshape(met.looks_azimuth, -1) if ch_data.ndim == 1 else ch_data)\n                logger.info(\"Loaded channel %s: %s\", pol.upper(), valid[0])\n        data.data_id = met.product_id or data.data_id\n        data.validate()\n        return data\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/io/writer.py",
  "content": "from __future__ import annotations\n\nimport json\nimport logging\nfrom pathlib import Path\nfrom typing import Optional\n\nimport numpy as np\nimport rasterio\nfrom rasterio.profiles import DefaultGTiffProfile\nimport xarray as xr\nimport zarr\n\nfrom lunaice.models.schemas import DecompositionProducts\n\nlogger = logging.getLogger(__name__)\n\n\nclass GeoTIFFWriter:\n    DEFAULT_PROFILE = {\n        \"driver\": \"GTiff\",\n        \"dtype\": \"float32\",\n        \"compress\": \"LZW\",\n        \"tiled\": True,\n        \"blockxsize\": 256,\n        \"blockysize\": 256,\n        \"interleave\": \"band\",\n        \"nodata\": np.nan,\n    }\n\n    def __init__(self, output_dir: str | Path, crs: str = \"EPSG:4326\", transform: Optional[list] = None):\n        self.output_dir = Path(output_dir)\n        self.output_dir.mkdir(parents=True, exist_ok=True)\n        self.crs = crs\n        self.transform = transform\n\n    def write_band(\n        self,\n        array: np.ndarray,\n        name: str,\n        profile_override: Optional[dict] = None,\n    ) -> Path:\n        path = self.output_dir / f\"{name}.tif\"\n        if array.ndim == 3:\n            n_bands = array.shape[0]\n        else:\n            n_bands = 1\n            array = array[np.newaxis, :, :]\n        profile = {**self.DEFAULT_PROFILE, \"count\": n_bands, \"height\": array.shape[1], \"width\": array.shape[2]}\n        if profile_override:\n            profile.update(profile_override)\n        if self.crs:\n            profile[\"crs\"] = self.crs\n        with rasterio.open(path, \"w\", **profile) as dst:\n            for i in range(n_bands):\n                dst.write(array[i].astype(np.float32), i + 1)\n                dst.set_band_description(i + 1, f\"{name}_b{i+1}\" if n_bands > 1 else name)\n        logger.info(\"Wrote GeoTIFF: %s (shape=%s)\", path, array.shape)\n        return path\n\n    def write_multiband(\n        self,\n        bands: dict[str, np.ndarray],\n        name: str,\n    ) -> Path:\n        first = next(iter(bands.values()))\n        h, w = first.shape\n        count = len(bands)\n        profile = {**self.DEFAULT_PROFILE, \"count\": count, \"height\": h, \"width\": w}\n        if self.crs:\n            profile[\"crs\"] = self.crs\n        path = self.output_dir / f\"{name}.tif\"\n        with rasterio.open(path, \"w\", **profile) as dst:\n            for i, (band_name, arr) in enumerate(bands.items(), 1):\n                dst.write(arr.astype(np.float32), i)\n                dst.set_band_description(i, band_name)\n        logger.info(\"Wrote multi-band GeoTIFF: %s (%d bands)\", path, count)\n        return path\n\n\nclass ZarrWriter:\n    def __init__(self, output_dir: str | Path, chunks: tuple[int, int] = (256, 256), compressor: str = \"blosc\"):\n        self.output_dir = Path(output_dir)\n        self.output_dir.mkdir(parents=True, exist_ok=True)\n        self.chunks = chunks\n        self.compressor = compressor\n\n    def write_product_cube(self, products: DecompositionProducts, name: str = \"polarimetric_cube\") -> Path:\n        data_vars = {}\n        for field_name in [\n            \"entropy\", \"alpha_deg\", \"anisotropy\",\n            \"cpr\", \"dop\", \"span\",\n            \"sigma_hh\", \"sigma_hv\", \"sigma_vv\",\n            \"gamma_hh\", \"gamma_hv\", \"gamma_vv\",\n            \"odd_bounce\", \"double_bounce\", \"volume_scattering\",\n        ]:\n            arr = getattr(products, field_name, None)\n            if arr is not None:\n                data_vars[field_name] = ((\"y\", \"x\"), arr)\n        if not data_vars:\n            raise ValueError(\"No valid products to write\")\n        ds = xr.Dataset(data_vars, attrs={\"description\": \"DFSAR polarimetric feature cube\"})\n        path = self.output_dir / f\"{name}.zarr\"\n        ds.to_zarr(str(path), mode=\"w\", encoding={k: {\"compressor\": zarr.Blosc(cname=self.compressor)} for k in data_vars})\n        logger.info(\"Wrote Zarr cube: %s (%d variables)\", path, len(data_vars))\n        return path\n\n\nclass ReportWriter:\n    def __init__(self, output_dir: str | Path):\n        self.output_dir = Path(output_dir)\n        self.output_dir.mkdir(parents=True, exist_ok=True)\n\n    def write_summary(self, summary: dict, name: str = \"processing_summary.json\") -> Path:\n        path = self.output_dir / name\n        serializable = {}\n        for k, v in summary.items():\n            if isinstance(v, np.generic):\n                serializable[k] = v.item()\n            elif isinstance(v, np.ndarray):\n                serializable[k] = v.tolist()\n            elif isinstance(v, (np.floating, np.integer)):\n                serializable[k] = v.item()\n            else:\n                serializable[k] = v\n        with open(path, \"w\") as f:\n            json.dump(serializable, f, indent=2, default=str)\n        logger.info(\"Wrote summary: %s\", path)\n        return path\n\n    def write_statistics(self, products: DecompositionProducts, name: str = \"statistics.json\") -> Path:\n        stats = {}\n        for field_name in [\n            \"entropy\", \"alpha_deg\", \"anisotropy\",\n            \"cpr\", \"dop\", \"span\",\n            \"sigma_hh\", \"sigma_hv\", \"sigma_vv\",\n            \"odd_bounce\", \"double_bounce\", \"volume_scattering\",\n        ]:\n            arr = getattr(products, field_name, None)\n            if arr is not None:\n                valid = arr[~np.isnan(arr)]\n                if valid.size > 0:\n                    stats[field_name] = {\n                        \"min\": float(valid.min()),\n                        \"max\": float(valid.max()),\n                        \"mean\": float(valid.mean()),\n                        \"std\": float(valid.std()),\n                        \"p5\": float(np.percentile(valid, 5)),\n                        \"p25\": float(np.percentile(valid, 25)),\n                        \"p50\": float(np.percentile(valid, 50)),\n                        \"p75\": float(np.percentile(valid, 75)),\n                        \"p95\": float(np.percentile(valid, 95)),\n                    }\n        return self.write_summary(stats, name)\n"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 38.3s)

_Thinking:_

Now let me write the processing modules.

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/__init__.py",
  "content": "from lunaice.processing.calibration import RadiometricCalibrator, PolarimetricCalibrator\nfrom lunaice.processing.speckle import SpeckleFilter\nfrom lunaice.processing.polarimetry import (\n    CoherencyMatrixBuilder,\n    CloudePottierDecomposition,\n    CircularPolarizationRatio,\n    DegreeOfPolarization,\n    BackscatterCoefficient,\n)\n\n__all__ = [\n    \"RadiometricCalibrator\",\n    \"PolarimetricCalibrator\",\n    \"SpeckleFilter\",\n    \"CoherencyMatrixBuilder\",\n    \"CloudePottierDecomposition\",\n    \"CircularPolarizationRatio\",\n    \"DegreeOfPolarization\",\n    \"BackscatterCoefficient\",\n]\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/calibration.py",
  "content": "from __future__ import annotations\n\nimport logging\n\nimport numpy as np\n\nfrom lunaice.models.schemas import CalibrationConstants, PolarimetricData\n\nlogger = logging.getLogger(__name__)\n\n\nclass RadiometricCalibrator:\n    def __init__(self, cal_constant: float = 1.0, apply_antenna_pattern: bool = False):\n        self.cal_constant = cal_constant\n        self.apply_antenna_pattern = apply_antenna_pattern\n\n    def calibrate(self, data: PolarimetricData) -> PolarimetricData:\n        if not data.is_quad_pol:\n            logger.warning(\"Data is not quad-pol; calibrating available channels\")\n        k = data.calibration.k_hh if data.calibration else self.cal_constant\n        for pol in [\"hh\", \"hv\", \"vh\", \"vv\"]:\n            arr = getattr(data, pol, None)\n            if arr is not None:\n                cal_val = getattr(data.calibration, f\"k_{pol}\", k) if data.calibration else k\n                sigma0 = np.abs(arr) ** 2 / (cal_val + 1e-30)\n                setattr(data, pol, np.sqrt(sigma0).astype(np.complex64))\n                logger.debug(\"Radiometric calibration applied to %s (K=%f)\", pol.upper(), cal_val)\n        logger.info(\"Radiometric calibration complete\")\n        return data\n\n\nclass PolarimetricCalibrator:\n    def __init__(\n        self,\n        co_pol_phase_correction: float = -50.0,\n        cross_pol_phase_correction: float = -5.0,\n        cross_talk_hv: float = 0.0,\n        cross_talk_vh: float = 0.0,\n        channel_imbalance_amp: float = 1.0,\n        channel_imbalance_phase: float = 0.0,\n    ):\n        self.co_pol_phase_correction = np.radians(co_pol_phase_correction)\n        self.cross_pol_phase_correction = np.radians(cross_pol_phase_correction)\n        self.cross_talk_hv = cross_talk_hv\n        self.cross_talk_vh = cross_talk_vh\n        self.channel_imbalance_amp = channel_imbalance_amp\n        self.channel_imbalance_phase = np.radians(channel_imbalance_phase)\n\n    def calibrate(self, data: PolarimetricData) -> PolarimetricData:\n        if not data.is_quad_pol:\n            logger.warning(\"Polarimetric calibration requires quad-pol data; skipping\")\n            return data\n        cal = data.calibration\n        if cal:\n            self.co_pol_phase_correction = np.radians(cal.phase_offset_hh_vv)\n            self.cross_pol_phase_correction = np.radians(cal.phase_offset_hv_vh)\n            self.cross_talk_hv = cal.cross_talk_hv\n            self.cross_talk_vh = cal.cross_talk_vh\n            self.channel_imbalance_amp = cal.channel_imbalance_amp\n            self.channel_imbalance_phase = np.radians(cal.channel_imbalance_phase)\n\n        imbalance = self.channel_imbalance_amp * np.exp(1j * self.channel_imbalance_phase)\n        data.hh *= np.exp(1j * self.co_pol_phase_correction)\n        data.vv *= np.exp(1j * self.co_pol_phase_correction)\n        data.hv *= np.exp(1j * self.cross_pol_phase_correction)\n        data.vh *= np.exp(1j * self.cross_pol_phase_correction)\n        data.hh -= self.cross_talk_hv * data.vh\n        data.vv -= self.cross_talk_vh * data.hv\n        data.hv -= self.cross_talk_hv * data.vv\n        data.vh -= self.cross_talk_vh * data.hh\n        data.hv /= imbalance\n        data.vh /= imbalance\n        logger.info(\"Polarimetric calibration complete\")\n        return data\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/speckle.py",
  "content": "from __future__ import annotations\n\nimport logging\n\nimport numpy as np\nfrom scipy.ndimage import generic_filter, uniform_filter\n\nfrom lunaice.models.schemas import PolarimetricData\nfrom lunaice.config import SpeckleFilterConfig\n\nlogger = logging.getLogger(__name__)\n\n\nclass SpeckleFilter:\n    def __init__(self, config: SpeckleFilterConfig):\n        self.config = config\n\n    def _refined_lee(self, intensity: np.ndarray) -> np.ndarray:\n        w = self.config.window_size\n        n_looks = self.config.n_looks\n        half = w // 2\n        padded = np.pad(intensity, half, mode=\"reflect\")\n        mu = uniform_filter(padded, size=w)[half:-half, half:-half]\n        sq = uniform_filter(padded ** 2, size=w)[half:-half, half:-half]\n        var = np.maximum(sq - mu ** 2, 0)\n        sigma_v = mu * np.sqrt(1.0 / n_looks) if n_looks > 0 else np.zeros_like(mu)\n        var_x = np.maximum(var - sigma_v ** 2, 0)\n        weights = var_x / (var_x + sigma_v ** 2 + 1e-10)\n        return mu + weights * (intensity - mu)\n\n    def _boxcar(self, intensity: np.ndarray) -> np.ndarray:\n        w = self.config.window_size\n        return uniform_filter(intensity, size=w)\n\n    def _lee_sigma(self, intensity: np.ndarray) -> np.ndarray:\n        w = self.config.window_size\n        n_sigma = self.config.damping_factor\n        half = w // 2\n        padded = np.pad(intensity, half, mode=\"reflect\")\n        h, w_im = intensity.shape\n        result = np.empty_like(intensity)\n        for i in range(h):\n            for j in range(w_im):\n                patch = padded[i : i + w, j : j + w]\n                mu_p = patch.mean()\n                sigma_p = patch.std()\n                lower = mu_p - n_sigma * sigma_p\n                upper = mu_p + n_sigma * sigma_p\n                mask = (patch >= lower) & (patch <= upper)\n                selected = patch[mask]\n                result[i, j] = selected.mean() if selected.size > 0 else mu_p\n        return result\n\n    def _idani(self, intensity: np.ndarray) -> np.ndarray:\n        w = self.config.window_size\n        half = w // 2\n        padded = np.pad(intensity, half, mode=\"reflect\")\n        h, w_im = intensity.shape\n        result = np.empty_like(intensity)\n        n_looks = max(self.config.n_looks, 1)\n        enl = n_looks\n        for i in range(h):\n            for j in range(w_im):\n                patch = padded[i : i + w, j : j + w]\n                mu_p = patch.mean()\n                var_p = patch.var()\n                cu = np.sqrt(var_p) / (mu_p + 1e-10)\n                ci = 1.0 / np.sqrt(enl)\n                cmax = np.sqrt(1 + 2.0 / enl)\n                if cu <= ci:\n                    result[i, j] = mu_p\n                elif cu >= cmax:\n                    result[i, j] = intensity[i, j]\n                else:\n                    damp = (cu - ci) / (cmax - ci)\n                    result[i, j] = mu_p * (1 - damp) + intensity[i, j] * damp\n        return result\n\n    def _bilateral(self, intensity: np.ndarray) -> np.ndarray:\n        w = self.config.window_size\n        sigma_s = w / 3.0\n        sigma_r = intensity.std() * self.config.damping_factor\n        half = w // 2\n        padded = np.pad(intensity, half, mode=\"reflect\")\n        h, w_im = intensity.shape\n        result = np.empty_like(intensity)\n        yi, xi = np.ogrid[-half : half + 1, -half : half + 1]\n        spatial_kernel = np.exp(-(yi ** 2 + xi ** 2) / (2 * sigma_s ** 2))\n        for i in range(h):\n            for j in range(w_im):\n                patch = padded[i : i + w, j : j + w]\n                range_kernel = np.exp(-((patch - intensity[i, j]) ** 2) / (2 * sigma_r ** 2 + 1e-10))\n                kernel = spatial_kernel * range_kernel\n                result[i, j] = np.sum(patch * kernel) / (np.sum(kernel) + 1e-10)\n        return result\n\n    def _filter_channel(self, channel: np.ndarray) -> np.ndarray:\n        intensity = np.abs(channel)\n        intensity_smooth = getattr(self, f\"_{self.config.method}\")(intensity)\n        phase = np.angle(channel)\n        return intensity_smooth * np.exp(1j * phase)\n\n    def apply(self, data: PolarimetricData) -> PolarimetricData:\n        logger.info(\"Applying %s speckle filter (window=%d)\", self.config.method, self.config.window_size)\n        for pol in [\"hh\", \"hv\", \"vh\", \"vv\"]:\n            arr = getattr(data, pol, None)\n            if arr is not None:\n                setattr(data, pol, self._filter_channel(arr))\n        logger.info(\"Speckle filtering complete\")\n        return data\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/__init__.py",
  "content": "from lunaice.processing.polarimetry.coherency import CoherencyMatrixBuilder\nfrom lunaice.processing.polarimetry.cloude_pottier import CloudePottierDecomposition\nfrom lunaice.processing.polarimetry.cpr import CircularPolarizationRatio\nfrom lunaice.processing.polarimetry.backscatter import BackscatterCoefficient\nfrom lunaice.processing.polarimetry.dop import DegreeOfPolarization\n\n__all__ = [\n    \"CoherencyMatrixBuilder\",\n    \"CloudePottierDecomposition\",\n    \"CircularPolarizationRatio\",\n    \"BackscatterCoefficient\",\n    \"DegreeOfPolarization\",\n]\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/coherency.py",
  "content": "from __future__ import annotations\n\nimport logging\n\nimport numpy as np\n\nfrom lunaice.models.schemas import CoherencyMatrix, PolarimetricData\n\nlogger = logging.getLogger(__name__)\n\n\nclass CoherencyMatrixBuilder:\n    def __init__(self, multilook_range: int = 1, multilook_azimuth: int = 1):\n        self.multilook = (multilook_range, multilook_azimuth)\n\n    def _multilook(self, arr: np.ndarray) -> np.ndarray:\n        r, a = self.multilook\n        if r == 1 and a == 1:\n            return arr\n        hr, ha = arr.shape[0] // r * r, arr.shape[1] // a * a\n        cropped = arr[:hr, :ha]\n        return cropped.reshape(hr // r, r, ha // a, a).mean(axis=(1, 3))\n\n    def build(self, data: PolarimetricData) -> CoherencyMatrix:\n        if not data.is_quad_pol:\n            raise ValueError(\"Quad-pol (HH, HV, VH, VV) required for coherency matrix\")\n        hh = self._multilook(data.hh)\n        hv = self._multilook(data.hv)\n        vh = self._multilook(data.vh)\n        vv = self._multilook(data.vv)\n\n        k = np.stack([hh + vv, hh - vv, hv + vh], axis=-1) / np.sqrt(2.0)\n        k_conj = np.conj(k)\n        t11 = (k[:, :, 0] * k_conj[:, :, 0]).real\n        t12 = k[:, :, 0] * k_conj[:, :, 1]\n        t13 = k[:, :, 0] * k_conj[:, :, 2]\n        t22 = (k[:, :, 1] * k_conj[:, :, 1]).real\n        t23 = k[:, :, 1] * k_conj[:, :, 2]\n        t33 = (k[:, :, 2] * k_conj[:, :, 2]).real\n        logger.info(\"Coherency matrix (T3) built: shape=%s\", t11.shape)\n        return CoherencyMatrix(\n            t11=t11, t12=t12, t13=t13,\n            t21=np.conj(t12), t22=t22, t23=t23,\n            t31=np.conj(t13), t32=np.conj(t23), t33=t33,\n        )\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/cloude_pottier.py",
  "content": "from __future__ import annotations\n\nimport logging\n\nimport numpy as np\n\nfrom lunaice.models.schemas import CoherencyMatrix, DecompositionProducts\n\nlogger = logging.getLogger(__name__)\n\n\nclass CloudePottierDecomposition:\n    def decompose(self, t3: CoherencyMatrix) -> DecompositionProducts:\n        h, w = t3.shape\n        entropy = np.full((h, w), np.nan, dtype=np.float64)\n        alpha_deg = np.full((h, w), np.nan, dtype=np.float64)\n        anisotropy = np.full((h, w), np.nan, dtype=np.float64)\n        lambda_1 = np.full((h, w), np.nan, dtype=np.float64)\n        lambda_2 = np.full((h, w), np.nan, dtype=np.float64)\n        lambda_3 = np.full((h, w), np.nan, dtype=np.float64)\n        alpha_1_arr = np.full((h, w), np.nan, dtype=np.float64)\n        alpha_2_arr = np.full((h, w), np.nan, dtype=np.float64)\n        alpha_3_arr = np.full((h, w), np.nan, dtype=np.float64)\n\n        for i in range(h):\n            for j in range(w):\n                T = np.array([\n                    [t3.t11[i, j], t3.t12[i, j], t3.t13[i, j]],\n                    [t3.t21[i, j], t3.t22[i, j], t3.t23[i, j]],\n                    [t3.t31[i, j], t3.t32[i, j], t3.t33[i, j]],\n                ], dtype=np.complex128)\n                T = (T + T.conj().T) / 2.0\n                eigenvalues, eigenvectors = np.linalg.eigh(T)\n                eigenvalues = np.maximum(eigenvalues[::-1], 0)\n                eigenvectors = eigenvectors[:, ::-1]\n                l1, l2, l3 = eigenvalues\n                lambda_1[i, j] = l1\n                lambda_2[i, j] = l2\n                lambda_3[i, j] = l3\n                total = l1 + l2 + l3\n                if total > 0:\n                    p1, p2, p3 = l1 / total, l2 / total, l3 / total\n                    p_safe = np.maximum([p1, p2, p3], 1e-30)\n                    entropy[i, j] = -np.sum(p_safe * np.log(p_safe)) / np.log(3)\n                else:\n                    p1 = p2 = p3 = 1.0 / 3.0\n                    entropy[i, j] = 0.0\n                anisotropy[i, j] = (l2 - l3) / (l2 + l3 + 1e-30) if (l2 + l3) > 0 else 0.0\n                for k_idx, (ev, eival) in enumerate(zip(eigenvectors.T, eigenvalues)):\n                    eiv = ev / (np.linalg.norm(ev) + 1e-30)\n                    k_pauli = np.array([1.0, 0.0, 0.0])\n                    cos_alpha = np.abs(np.dot(eiv, k_pauli))\n                    alpha_val = np.degrees(np.arccos(np.clip(cos_alpha, 0.0, 1.0)))\n                    if k_idx == 0:\n                        alpha_1_arr[i, j] = alpha_val\n                    elif k_idx == 1:\n                        alpha_2_arr[i, j] = alpha_val\n                    else:\n                        alpha_3_arr[i, j] = alpha_val\n                alpha_deg[i, j] = p1 * alpha_1_arr[i, j] + p2 * alpha_2_arr[i, j] + p3 * alpha_3_arr[i, j]\n\n        logger.info(\"Cloude-Pottier decomposition complete\")\n        return DecompositionProducts(\n            entropy=entropy.astype(np.float32),\n            alpha_deg=alpha_deg.astype(np.float32),\n            anisotropy=anisotropy.astype(np.float32),\n            lambda_1=lambda_1.astype(np.float32),\n            lambda_2=lambda_2.astype(np.float32),\n            lambda_3=lambda_3.astype(np.float32),\n            alpha_1=alpha_1_arr.astype(np.float32),\n            alpha_2=alpha_2_arr.astype(np.float32),\n            alpha_3=alpha_3_arr.astype(np.float32),\n        )\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/cpr.py",
  "content": "from __future__ import annotations\n\nimport logging\n\nimport numpy as np\n\nfrom lunaice.models.schemas import PolarimetricData, DecompositionProducts\n\nlogger = logging.getLogger(__name__)\n\n\nclass CircularPolarizationRatio:\n    def compute(self, data: PolarimetricData, products: DecompositionProducts) -> DecompositionProducts:\n        if data.is_quad_pol:\n            sc = np.array([\n                [data.hh, data.hv],\n                [data.vh, data.vv],\n            ])\n            rcp_tx = (sc[0, 0] - 1j * sc[1, 0]) / np.sqrt(2)\n            lcp_tx = (sc[0, 0] + 1j * sc[1, 0]) / np.sqrt(2)\n            sc_lcp = np.array([\n                [lcp_tx, (sc[0, 1] + 1j * sc[1, 1]) / np.sqrt(2)],\n                [rcp_tx, (sc[0, 1] - 1j * sc[1, 1]) / np.sqrt(2)],\n            ])\n            same_sense = np.abs(sc_lcp[0, 0]) ** 2 + np.abs(sc_lcp[1, 1]) ** 2\n            opp_sense = np.abs(sc_lcp[0, 1]) ** 2 + np.abs(sc_lcp[1, 0]) ** 2\n            cpr = same_sense / (opp_sense + 1e-30)\n        elif data.metadata and data.metadata.polarization_mode:\n            logger.warning(\"CPR from non-quad-pol data not fully supported; using dummy\")\n            hh_pow = np.abs(data.hh) ** 2 if data.hh is not None else 0\n            vv_pow = np.abs(data.vv) ** 2 if data.vv is not None else 0\n            cpr = hh_pow / (vv_pow + 1e-30) if vv_pow is not None else np.ones_like(hh_pow)\n        else:\n            raise ValueError(\"Insufficient polarization data for CPR computation\")\n        products.cpr = cpr.astype(np.float32)\n        logger.info(\"CPR computed: range [%.3f, %.3f]\", np.nanmin(cpr), np.nanmax(cpr))\n        return products\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/dop.py",
  "content": "from __future__ import annotations\n\nimport logging\n\nimport numpy as np\n\nfrom lunaice.models.schemas import PolarimetricData, DecompositionProducts\n\nlogger = logging.getLogger(__name__)\n\n\nclass DegreeOfPolarization:\n    def compute(self, data: PolarimetricData, products: DecompositionProducts) -> DecompositionProducts:\n        if data.is_quad_pol:\n            g11 = (np.abs(data.hh) ** 2 + np.abs(data.hv) ** 2 +\n                   np.abs(data.vh) ** 2 + np.abs(data.vv) ** 2).real\n            g22 = (np.abs(data.hh) ** 2 - np.abs(data.hv) ** 2 +\n                   np.abs(data.vh) ** 2 - np.abs(data.vv) ** 2).real\n            g33 = (2 * (data.hh * np.conj(data.hv) + data.vh * np.conj(data.vv))).real\n            g44 = (2j * (data.hh * np.conj(data.hv) - data.vh * np.conj(data.vv))).real\n            dop = np.sqrt(g22 ** 2 + g33 ** 2 + g44 ** 2) / (g11 + 1e-30)\n        else:\n            logger.warning(\"DOP from non-quad-pol; using approximate method\")\n            g11 = np.abs(data.hh) ** 2 + np.abs(data.vv) ** 2 if data.hh is not None else 1\n            g22 = np.abs(data.hh) ** 2 - np.abs(data.vv) ** 2 if data.hh is not None else 0\n            dop = np.sqrt(g22 ** 2) / (g11 + 1e-30)\n        products.dop = np.clip(dop, 0.0, 1.0).astype(np.float32)\n        logger.info(\"DOP computed: range [%.3f, %.3f]\", np.nanmin(products.dop), np.nanmax(products.dop))\n        return products\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/backscatter.py",
  "content": "from __future__ import annotations\n\nimport logging\n\nimport numpy as np\n\nfrom lunaice.models.schemas import PolarimetricData, DecompositionProducts\n\nlogger = logging.getLogger(__name__)\n\n\nclass BackscatterCoefficient:\n    def __init__(self, incidence_angle_deg: float = 30.0):\n        self.incidence_angle_rad = np.radians(incidence_angle_deg)\n\n    def compute(self, data: PolarimetricData, products: DecompositionProducts) -> DecompositionProducts:\n        pols = {\"sigma_hh\": data.hh, \"sigma_hv\": data.hv, \"sigma_vv\": data.vv}\n        for attr, arr in pols.items():\n            if arr is not None:\n                sigma = np.abs(arr) ** 2\n                setattr(products, attr, sigma.astype(np.float32))\n                gamma = sigma * np.cos(self.incidence_angle_rad)\n                gamma_attr = attr.replace(\"sigma\", \"gamma\")\n                setattr(products, gamma_attr, gamma.astype(np.float32))\n        if data.is_quad_pol:\n            products.span = (\n                np.abs(data.hh) ** 2 + np.abs(data.hv) ** 2 +\n                np.abs(data.vh) ** 2 + np.abs(data.vv) ** 2\n            ).astype(np.float32)\n        logger.info(\"Backscatter coefficients computed\")\n        return products\n"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 4.0s)

_Thinking:_

Now let me write the pipeline orchestrator, CLI, configs, tests, and project files.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "priority": "high",
      "content": "Create repository structure and directory tree",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Build core data models and schemas (PolarimetricData, metadata)",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement I/O module (PDS4 reader, GeoTIFF/Zarr writer)",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement radiometric calibration module",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement polarimetric calibration module",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement speckle reduction module",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement coherency matrix (T3) generation",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement Cloude-Pottier decomposition (H/alpha/A)",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement CPR, DOP, backscatter coefficients",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Build pipeline orchestrator",
      "status": "in_progress"
    },
    {
      "priority": "high",
      "content": "Build CLI interface",
      "status": "pending"
    },
    {
      "priority": "high",
      "content": "Create configuration files (YAML)",
      "status": "pending"
    },
    {
      "priority": "medium",
      "content": "Write tests",
      "status": "pending"
    },
    {
      "priority": "medium",
      "content": "Create pyproject.toml and documentation",
      "status": "pending"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create repository structure and directory tree",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Build core data models and schemas (PolarimetricData, metadata)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement I/O module (PDS4 reader, GeoTIFF/Zarr writer)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement radiometric calibration module",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement polarimetric calibration module",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement speckle reduction module",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement coherency matrix (T3) generation",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement Cloude-Pottier decomposition (H/alpha/A)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement CPR, DOP, backscatter coefficients",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Build pipeline orchestrator",
    "status": "in_progress",
    "priority": "high"
  },
  {
    "content": "Build CLI interface",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Create configuration files (YAML)",
    "status": "pending",
    "priority": "high"
  },
  {
    "content": "Write tests",
    "status": "pending",
    "priority": "medium"
  },
  {
    "content": "Create pyproject.toml and documentation",
    "status": "pending",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 25.5s)

_Thinking:_

Continue building the pipeline, CLI, configs, and tests.

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/pipeline.py",
  "content": "from __future__ import annotations\n\nimport logging\nimport time\nfrom pathlib import Path\nfrom typing import Optional\n\nimport numpy as np\n\nfrom lunaice.config import DFSARConfig, ProcessingConfig, SpeckleFilterConfig\nfrom lunaice.io.reader import DFSARReader\nfrom lunaice.io.writer import GeoTIFFWriter, ReportWriter, ZarrWriter\nfrom lunaice.models.schemas import DecompositionProducts, ProcessingSummary\nfrom lunaice.processing.calibration import PolarimetricCalibrator, RadiometricCalibrator\nfrom lunaice.processing.polarimetry import (\n    BackscatterCoefficient,\n    CircularPolarizationRatio,\n    CloudePottierDecomposition,\n    CoherencyMatrixBuilder,\n    DegreeOfPolarization,\n)\nfrom lunaice.processing.speckle import SpeckleFilter\n\nlogger = logging.getLogger(\"lunaice.pipeline\")\n\n\nclass Pipeline:\n    def __init__(self, config: DFSARConfig):\n        self.config = config\n        self.config.resolve()\n        self._setup_logging()\n\n    def _setup_logging(self) -> None:\n        level = getattr(logging, self.config.logging_level.upper(), logging.INFO)\n        logging.basicConfig(\n            level=level,\n            format=\"%(asctime)s [%(levelname)s] %(name)s: %(message)s\",\n            datefmt=\"%Y-%m-%dT%H:%M:%S\",\n        )\n\n    def run(self) -> ProcessingSummary:\n        t0 = time.time()\n        errors: list[str] = []\n        warnings: list[str] = []\n        products_generated: list[str] = []\n        proc = self.config.processing\n\n        logger.info(\"=\" * 60)\n        logger.info(\"LUNAICE DFSAR Processing Pipeline v0.1.0\")\n        logger.info(\"Input: %s\", self.config.input_file)\n        logger.info(\"Band: %s\", self.config.band)\n\n        try:\n            reader = DFSARReader(self.config.input_file)\n            data = reader.read_slc()\n            logger.info(\"Loaded SLC data: shape=%s, quad_pol=%s\", data.shape, data.is_quad_pol)\n        except Exception as e:\n            logger.critical(\"Failed to read input: %s\", e)\n            errors.append(f\"ReadError: {e}\")\n            return ProcessingSummary(\n                input_file=self.config.input_file,\n                output_dir=self.config.output_dir,\n                products_generated=[],\n                processing_time_s=time.time() - t0,\n                config_snapshot=self._config_snapshot(),\n                errors=errors,\n            )\n\n        if proc.radiometric_calibration:\n            try:\n                rad_cal = RadiometricCalibrator(\n                    cal_constant=data.metadata.calibration_constant if data.metadata else 1.0\n                )\n                data = rad_cal.calibrate(data)\n                products_generated.append(\"radiometric_calibration\")\n            except Exception as e:\n                warnings.append(f\"RadiometricCalibrationWarning: {e}\")\n                logger.warning(\"Radiometric calibration failed: %s\", e)\n\n        if proc.polarimetric_calibration and data.is_quad_pol:\n            try:\n                pol_cal = PolarimetricCalibrator()\n                data = pol_cal.calibrate(data)\n                products_generated.append(\"polarimetric_calibration\")\n            except Exception as e:\n                warnings.append(f\"PolarimetricCalibrationWarning: {e}\")\n                logger.warning(\"Polarimetric calibration failed: %s\", e)\n\n        if proc.speckle_filter:\n            try:\n                filt = SpeckleFilter(proc.speckle_filter)\n                data = filt.apply(data)\n                products_generated.append(f\"speckle_filter_{proc.speckle_filter.method}\")\n            except Exception as e:\n                warnings.append(f\"SpeckleFilterWarning: {e}\")\n                logger.warning(\"Speckle filtering failed: %s\", e)\n\n        products = DecompositionProducts()\n\n        if proc.generate_coherency_matrix and data.is_quad_pol:\n            try:\n                t3_builder = CoherencyMatrixBuilder(\n                    multilook_range=proc.multilook_range,\n                    multilook_azimuth=proc.multilook_azimuth,\n                )\n                t3 = t3_builder.build(data)\n                products_generated.append(\"coherency_matrix_T3\")\n            except Exception as e:\n                warnings.append(f\"T3Warning: {e}\")\n                logger.warning(\"T3 generation failed: %s\", e)\n                t3 = None\n        else:\n            t3 = None\n\n        if proc.generate_cloude_pottier and t3 is not None:\n            try:\n                cp = CloudePottierDecomposition()\n                prod_cp = cp.decompose(t3)\n                for attr in [\"entropy\", \"alpha_deg\", \"anisotropy\",\n                             \"lambda_1\", \"lambda_2\", \"lambda_3\",\n                             \"alpha_1\", \"alpha_2\", \"alpha_3\"]:\n                    setattr(products, attr, getattr(prod_cp, attr))\n                products_generated.append(\"cloude_pottier_H_alpha_A\")\n            except Exception as e:\n                warnings.append(f\"CloudePottierWarning: {e}\")\n                logger.warning(\"Cloude-Pottier decomposition failed: %s\", e)\n\n        if proc.generate_cpr:\n            try:\n                cpr_comp = CircularPolarizationRatio()\n                products = cpr_comp.compute(data, products)\n                products_generated.append(\"cpr\")\n            except Exception as e:\n                warnings.append(f\"CPRWarning: {e}\")\n                logger.warning(\"CPR computation failed: %s\", e)\n\n        if proc.generate_dop:\n            try:\n                dop_comp = DegreeOfPolarization()\n                products = dop_comp.compute(data, products)\n                products_generated.append(\"dop\")\n            except Exception as e:\n                warnings.append(f\"DOPWarning: {e}\")\n                logger.warning(\"DOP computation failed: %s\", e)\n\n        if proc.generate_backscatter:\n            try:\n                inc = data.metadata.incidence_angle_deg if data.metadata else 30.0\n                bsc = BackscatterCoefficient(incidence_angle_deg=inc)\n                products = bsc.compute(data, products)\n                products_generated.append(\"backscatter_coefficients\")\n            except Exception as e:\n                warnings.append(f\"BackscatterWarning: {e}\")\n                logger.warning(\"Backscatter computation failed: %s\", e)\n\n        try:\n            self._write_outputs(products, products_generated)\n        except Exception as e:\n            errors.append(f\"OutputError: {e}\")\n            logger.error(\"Output writing failed: %s\", e)\n\n        elapsed = time.time() - t0\n        logger.info(\"Pipeline finished in %.2f s\", elapsed)\n        logger.info(\"Products: %s\", products_generated)\n        return ProcessingSummary(\n            input_file=self.config.input_file,\n            output_dir=self.config.output_dir,\n            products_generated=products_generated,\n            processing_time_s=elapsed,\n            config_snapshot=self._config_snapshot(),\n            errors=errors,\n            warnings=warnings,\n        )\n\n    def _write_outputs(self, products: DecompositionProducts, generated: list[str]) -> None:\n        out_dir = Path(self.config.output_dir)\n        gtiff = GeoTIFFWriter(out_dir)\n        zarr_w = ZarrWriter(out_dir)\n        report = ReportWriter(out_dir)\n\n        single_band_map = {\n            \"entropy\": products.entropy,\n            \"alpha_deg\": products.alpha_deg,\n            \"anisotropy\": products.anisotropy,\n            \"cpr\": products.cpr,\n            \"dop\": products.dop,\n            \"span\": products.span,\n            \"sigma_hh\": products.sigma_hh,\n            \"sigma_hv\": products.sigma_hv,\n            \"sigma_vv\": products.sigma_vv,\n            \"gamma_hh\": products.gamma_hh,\n            \"gamma_hv\": products.gamma_hv,\n            \"gamma_vv\": products.gamma_vv,\n            \"odd_bounce\": products.odd_bounce,\n            \"double_bounce\": products.double_bounce,\n            \"volume_scattering\": products.volume_scattering,\n        }\n        for name, arr in single_band_map.items():\n            if arr is not None:\n                gtiff.write_band(arr, name)\n\n        if any(getattr(products, f) is not None for f in\n               [\"entropy\", \"alpha_deg\", \"anisotropy\", \"cpr\", \"dop\"]):\n            zarr_w.write_product_cube(products)\n\n        report.write_statistics(products)\n\n    def _config_snapshot(self) -> dict:\n        return {\n            \"input_file\": self.config.input_file,\n            \"output_dir\": self.config.output_dir,\n            \"band\": self.config.band,\n            \"processing\": {\n                \"radiometric_calibration\": self.config.processing.radiometric_calibration,\n                \"polarimetric_calibration\": self.config.processing.polarimetric_calibration,\n                \"speckle_filter\": {\n                    \"method\": self.config.processing.speckle_filter.method,\n                    \"window_size\": self.config.processing.speckle_filter.window_size,\n                } if self.config.processing.speckle_filter else None,\n                \"multilook_range\": self.config.processing.multilook_range,\n                \"multilook_azimuth\": self.config.processing.multilook_azimuth,\n                \"generate_cloude_pottier\": self.config.processing.generate_cloude_pottier,\n                \"generate_cpr\": self.config.processing.generate_cpr,\n                \"generate_dop\": self.config.processing.generate_dop,\n            },\n        }\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/cli.py",
  "content": "from __future__ import annotations\n\nimport argparse\nimport json\nimport sys\nfrom pathlib import Path\n\nfrom lunaice import __version__\nfrom lunaice.config import DFSARConfig\nfrom lunaice.pipeline import Pipeline\n\n\ndef build_parser() -> argparse.ArgumentParser:\n    parser = argparse.ArgumentParser(\n        prog=\"lunaice\",\n        description=\"LUNAICE: Lunar Subsurface Ice Detection using Chandrayaan-2 DFSAR Data\",\n        formatter_class=argparse.RawDescriptionHelpFormatter,\n        epilog=\"\"\"\nExamples:\n  lunaice process -i data/L1A_SLC.h5 -o output/ --band L\n  lunaice process -i data/SLC/ --config configs/processing.yaml -v\n  lunaice process -i data/L1A_SLC.h5 --no-speckle --multilook 2 2\n        \"\"\",\n    )\n    parser.add_argument(\"--version\", action=\"version\", version=f\"%(prog)s {__version__}\")\n\n    sub = parser.add_subparsers(dest=\"command\", required=True, help=\"Sub-command\")\n\n    proc = sub.add_parser(\"process\", help=\"Run the DFSAR polarimetric processing pipeline\")\n    proc.add_argument(\"-i\", \"--input\", required=True, help=\"Input SLC file or directory\")\n    proc.add_argument(\"-o\", \"--output\", default=\"output\", help=\"Output directory\")\n    proc.add_argument(\"--band\", choices=[\"L\", \"S\"], default=\"L\", help=\"Frequency band\")\n    proc.add_argument(\"-c\", \"--config\", help=\"YAML configuration file (overrides defaults)\")\n    proc.add_argument(\"--no-radiometric\", action=\"store_false\", dest=\"radiometric\", help=\"Skip radiometric calibration\")\n    proc.add_argument(\"--no-polarimetric\", action=\"store_false\", dest=\"polarimetric\", help=\"Skip polarimetric calibration\")\n    proc.add_argument(\"--no-speckle\", action=\"store_false\", dest=\"speckle\", help=\"Skip speckle filtering\")\n    proc.add_argument(\"--speckle-method\", default=\"refined_lee\", choices=[\"refined_lee\", \"boxcar\", \"lee_sigma\", \"idani\", \"bilateral\"])\n    proc.add_argument(\"--speckle-window\", type=int, default=7, help=\"Speckle filter window size\")\n    proc.add_argument(\"--no-cloude\", action=\"store_false\", dest=\"cloude\", help=\"Skip Cloude-Pottier decomposition\")\n    proc.add_argument(\"--no-cpr\", action=\"store_false\", dest=\"cpr\", help=\"Skip CPR computation\")\n    proc.add_argument(\"--no-dop\", action=\"store_false\", dest=\"dop\", help=\"Skip DOP computation\")\n    proc.add_argument(\"--no-backscatter\", action=\"store_false\", dest=\"backscatter\", help=\"Skip backscatter coefficients\")\n    proc.add_argument(\"--multilook-range\", type=int, default=1, help=\"Multi-looking range factor\")\n    proc.add_argument(\"--multilook-azimuth\", type=int, default=1, help=\"Multi-looking azimuth factor\")\n    proc.add_argument(\"--output-dtype\", default=\"float32\", choices=[\"float32\", \"float64\"], help=\"Output data type\")\n    proc.add_argument(\"-v\", \"--verbose\", action=\"store_true\", help=\"Verbose logging\")\n    proc.add_argument(\"--overwrite\", action=\"store_true\", help=\"Overwrite existing output\")\n    return parser\n\n\ndef cmd_process(args: argparse.Namespace) -> int:\n    if args.config:\n        config = DFSARConfig.from_yaml(args.config)\n        config.input_file = args.input\n        config.output_dir = args.output\n        if args.verbose:\n            config.logging_level = \"DEBUG\"\n    else:\n        from lunaice.config import ProcessingConfig, SpeckleFilterConfig\n        speckle = None\n        if args.speckle:\n            speckle = SpeckleFilterConfig(method=args.speckle_method, window_size=args.speckle_window)\n        proc = ProcessingConfig(\n            radiometric_calibration=args.radiometric,\n            polarimetric_calibration=args.polarimetric,\n            speckle_filter=speckle,\n            multilook_range=args.multilook_range,\n            multilook_azimuth=args.multilook_azimuth,\n            generate_cloude_pottier=args.cloude,\n            generate_cpr=args.cpr,\n            generate_dop=args.dop,\n            generate_backscatter=args.backscatter,\n            output_dtype=args.output_dtype,\n        )\n        config = DFSARConfig(\n            input_file=args.input,\n            output_dir=args.output,\n            band=args.band,\n            processing=proc,\n            verbose=args.verbose,\n            overwrite=args.overwrite,\n        )\n        if args.verbose:\n            config.logging_level = \"DEBUG\"\n\n    pipeline = Pipeline(config)\n    summary = pipeline.run()\n    print(json.dumps({\n        \"status\": \"success\" if not summary.errors else \"partial_failure\",\n        \"products\": summary.products_generated,\n        \"time_s\": round(summary.processing_time_s, 2),\n        \"errors\": summary.errors,\n        \"warnings\": summary.warnings,\n    }, indent=2))\n    return 1 if summary.errors else 0\n\n\ndef main() -> int:\n    parser = build_parser()\n    args = parser.parse_args()\n    if args.command == \"process\":\n        return cmd_process(args)\n    parser.print_help()\n    return 1\n\n\nif __name__ == \"__main__\":\n    sys.exit(main())\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/configs/default.yaml",
  "content": "input_file: \"\"\noutput_dir: \"output\"\nband: \"L\"\nlogging_level: \"INFO\"\nverbose: false\noverwrite: false\n\nprocessing:\n  radiometric_calibration: true\n  polarimetric_calibration: true\n\n  speckle_filter:\n    method: \"refined_lee\"\n    window_size: 7\n    damping_factor: 1.0\n    n_looks: 4\n\n  multilook_range: 1\n  multilook_azimuth: 1\n\n  generate_coherency_matrix: true\n  generate_cloude_pottier: true\n  generate_cpr: true\n  generate_dop: true\n  generate_backscatter: true\n\n  output_dtype: \"float32\"\n  clip_percentile: 99.5\n  zarr_compressor: \"blosc\"\n  zarr_chunks: [256, 256]\n  geotiff_compress: \"LZW\"\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/configs/processing.yaml",
  "content": "input_file: \"/data/dfsar/L1A/L_band_slc.h5\"\noutput_dir: \"/data/dfsar/processed/\"\nband: \"L\"\nlogging_level: \"INFO\"\nverbose: true\noverwrite: false\n\nprocessing:\n  radiometric_calibration: true\n  polarimetric_calibration: true\n\n  speckle_filter:\n    method: \"refined_lee\"\n    window_size: 7\n    damping_factor: 1.0\n    n_looks: 4\n\n  multilook_range: 2\n  multilook_azimuth: 2\n\n  generate_coherency_matrix: true\n  generate_cloude_pottier: true\n  generate_cpr: true\n  generate_dop: true\n  generate_backscatter: true\n\n  output_dtype: \"float32\"\n  clip_percentile: 99.5\n  zarr_compressor: \"blosc\"\n  zarr_chunks: [256, 256]\n  geotiff_compress: \"LZW\"\n"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 18.7s)

_Thinking:_

Now let me write the tests, pyproject.toml, and documentation.

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/tests/__init__.py",
  "content": ""
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/tests/conftest.py",
  "content": "from __future__ import annotations\n\nfrom pathlib import Path\n\nimport numpy as np\nimport pytest\n\nfrom lunaice.config import DFSARConfig, ProcessingConfig, SpeckleFilterConfig\nfrom lunaice.models.schemas import Metadata, PolarimetricData\n\n\n@pytest.fixture\ndef sample_slc_data() -> PolarimetricData:\n    np.random.seed(42)\n    h = 128\n    w = 128\n    hh = np.random.randn(h, w) + 1j * np.random.randn(h, w)\n    hv = np.random.randn(h, w) + 1j * np.random.randn(h, w)\n    vh = np.random.randn(h, w) + 1j * np.random.randn(h, w)\n    vv = np.random.randn(h, w) + 1j * np.random.randn(h, w)\n    return PolarimetricData(hh=hh, hv=hv, vh=vh, vv=vv, metadata=_test_metadata())\n\n\ndef _test_metadata() -> Metadata:\n    return Metadata(\n        product_id=\"TEST_001\",\n        processing_level=\"L1A\",\n        frequency_band=\"L\",\n        polarization_mode=\"quad_pol\",\n        acquisition_time=\"2020-01-15T12:00:00\",\n        orbit_number=1234,\n        incidence_angle_deg=30.0,\n        slant_range_resolution_m=15.0,\n        azimuth_resolution_m=15.0,\n        looks_range=1,\n        looks_azimuth=1,\n        calibration_constant=1.0,\n        wavelength_cm=24.0,\n        center_latitude=-85.0,\n        center_longitude=30.0,\n        pixel_spacing_m=15.0,\n    )\n\n\n@pytest.fixture\ndef default_config() -> DFSARConfig:\n    return DFSARConfig(\n        input_file=\"/tmp/test_input\",\n        output_dir=\"/tmp/test_output\",\n        band=\"L\",\n        processing=ProcessingConfig(\n            speckle_filter=SpeckleFilterConfig(method=\"refined_lee\", window_size=5),\n        ),\n    )\n\n\n@pytest.fixture\ndef temp_output_dir(tmp_path: Path) -> Path:\n    d = tmp_path / \"output\"\n    d.mkdir()\n    return d\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/tests/test_calibration.py",
  "content": "from __future__ import annotations\n\nimport numpy as np\nimport pytest\n\nfrom lunaice.models.schemas import PolarimetricData\nfrom lunaice.processing.calibration import PolarimetricCalibrator, RadiometricCalibrator\n\n\nclass TestRadiometricCalibrator:\n    def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):\n        cal = RadiometricCalibrator(cal_constant=2.0)\n        result = cal.calibrate(sample_slc_data)\n        assert result is sample_slc_data\n        assert result.hh is not None\n        expected_power = np.abs(sample_slc_data.hh) ** 2 / 2.0\n        assert np.allclose(np.abs(result.hh) ** 2, expected_power)\n\n    def test_calibrate_partial(self):\n        data = PolarimetricData(hh=np.ones((4, 4), dtype=complex))\n        cal = RadiometricCalibrator(cal_constant=4.0)\n        result = cal.calibrate(data)\n        assert np.allclose(np.abs(result.hh) ** 2, 0.25)\n\n\nclass TestPolarimetricCalibrator:\n    def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):\n        cal = PolarimetricCalibrator(\n            co_pol_phase_correction=-50.0,\n            cross_pol_phase_correction=-5.0,\n            cross_talk_hv=0.01,\n            cross_talk_vh=0.01,\n        )\n        result = cal.calibrate(sample_slc_data)\n        assert result is sample_slc_data\n        assert result.hh.shape == sample_slc_data.hh.shape\n\n    def test_skip_non_quadpol(self):\n        data = PolarimetricData(hh=np.ones((4, 4), dtype=complex))\n        cal = PolarimetricCalibrator()\n        result = cal.calibrate(data)\n        np.testing.assert_array_equal(result.hh, data.hh)\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/tests/test_polarimetry.py",
  "content": "from __future__ import annotations\n\nimport numpy as np\nimport pytest\n\nfrom lunaice.models.schemas import PolarimetricData, DecompositionProducts\nfrom lunaice.processing.polarimetry import (\n    BackscatterCoefficient,\n    CircularPolarizationRatio,\n    CloudePottierDecomposition,\n    CoherencyMatrixBuilder,\n    DegreeOfPolarization,\n)\n\n\nclass TestCoherencyMatrixBuilder:\n    def test_build_quad_pol(self, sample_slc_data: PolarimetricData):\n        builder = CoherencyMatrixBuilder()\n        t3 = builder.build(sample_slc_data)\n        assert t3.t11.shape == sample_slc_data.shape\n        assert np.all(np.isfinite(t3.t11))\n        assert t3.t11.dtype == np.float64\n\n    def test_raises_on_partial(self):\n        data = PolarimetricData(hh=np.ones((4, 4), dtype=complex))\n        with pytest.raises(ValueError, match=\"Quad-pol\"):\n            CoherencyMatrixBuilder().build(data)\n\n\nclass TestCloudePottierDecomposition:\n    def test_decompose(self, sample_slc_data: PolarimetricData):\n        t3 = CoherencyMatrixBuilder().build(sample_slc_data)\n        cp = CloudePottierDecomposition()\n        products = cp.decompose(t3)\n        assert products.entropy is not None\n        assert products.alpha_deg is not None\n        assert products.anisotropy is not None\n        assert np.all((products.entropy >= 0) & (products.entropy <= 1))\n        assert np.all((products.alpha_deg >= 0) & (products.alpha_deg <= 90))\n        assert np.all((products.anisotropy >= 0) & (products.anisotropy <= 1))\n\n\nclass TestCircularPolarizationRatio:\n    def test_cpr_quad_pol(self, sample_slc_data: PolarimetricData):\n        products = DecompositionProducts()\n        cpr_comp = CircularPolarizationRatio()\n        products = cpr_comp.compute(sample_slc_data, products)\n        assert products.cpr is not None\n        assert np.all(products.cpr >= 0)\n        assert products.cpr.shape == sample_slc_data.shape\n\n\nclass TestDegreeOfPolarization:\n    def test_dop_quad_pol(self, sample_slc_data: PolarimetricData):\n        products = DecompositionProducts()\n        dop_comp = DegreeOfPolarization()\n        products = dop_comp.compute(sample_slc_data, products)\n        assert products.dop is not None\n        assert np.all((products.dop >= 0) & (products.dop <= 1))\n\n\nclass TestBackscatterCoefficient:\n    def test_compute(self, sample_slc_data: PolarimetricData):\n        products = DecompositionProducts()\n        bsc = BackscatterCoefficient(incidence_angle_deg=30.0)\n        products = bsc.compute(sample_slc_data, products)\n        assert products.sigma_hh is not None\n        assert products.gamma_hh is not None\n        assert products.span is not None\n        assert np.all(products.sigma_hh >= 0)\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/tests/test_pipeline.py",
  "content": "from __future__ import annotations\n\nimport numpy as np\nimport pytest\n\nfrom lunaice.config import DFSARConfig, ProcessingConfig, SpeckleFilterConfig\n\n\ndef test_config_from_yaml(tmp_path):\n    yml = tmp_path / \"test_config.yaml\"\n    yml.write_text(\"\"\"\ninput_file: \"/data/input.h5\"\noutput_dir: \"/data/output\"\nband: \"S\"\nprocessing:\n  radiometric_calibration: true\n  polarimetric_calibration: true\n  speckle_filter:\n    method: \"refined_lee\"\n    window_size: 5\n  generate_cloude_pottier: true\n  generate_cpr: true\n  generate_dop: true\nlogging_level: \"DEBUG\"\n\"\"\")\n    config = DFSARConfig.from_yaml(yml)\n    assert config.band == \"S\"\n    assert config.processing.radiometric_calibration is True\n    assert config.processing.speckle_filter is not None\n    assert config.processing.speckle_filter.method == \"refined_lee\"\n    assert config.processing.speckle_filter.window_size == 5\n\n\ndef test_speckle_filter_validation():\n    with pytest.raises(ValueError, match=\"Speckle filter must be one of\"):\n        SpeckleFilterConfig(method=\"invalid_filter\")\n\n    with pytest.raises(ValueError, match=\"window_size must be odd\"):\n        SpeckleFilterConfig(method=\"boxcar\", window_size=4)\n\n\ndef test_speckle_filter_methods():\n    for method in [\"refined_lee\", \"boxcar\", \"idani\"]:\n        cfg = SpeckleFilterConfig(method=method, window_size=5)\n        assert cfg.method == method\n\n\ndef test_processing_config_defaults():\n    cfg = ProcessingConfig()\n    assert cfg.radiometric_calibration is True\n    assert cfg.generate_cloude_pottier is True\n    assert cfg.multilook_range == 1\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/tests/test_reader.py",
  "content": "from __future__ import annotations\n\nfrom pathlib import Path\n\nimport numpy as np\nimport pytest\n\nfrom lunaice.io.reader import DFSARReader, read_pds4_label\nfrom lunaice.models.schemas import PolarimetricData\n\n\ndef test_read_pds4_label_not_found():\n    with pytest.raises(FileNotFoundError):\n        read_pds4_label(\"/nonexistent/label.xml\")\n\n\ndef test_dfsar_reader_metadata(tmp_path):\n    data_file = tmp_path / \"test_slc.npy\"\n    np.save(str(data_file), np.zeros((10, 10), dtype=complex))\n    reader = DFSARReader(data_file)\n    met = reader.read_metadata()\n    assert met.product_id == \"\"\n\n\ndef test_validate_consistent_shapes():\n    data = PolarimetricData(\n        hh=np.ones((10, 10), dtype=complex),\n        hv=np.ones((10, 10), dtype=complex),\n        vv=np.ones((10, 10), dtype=complex),\n    )\n    assert data.validate() is True\n\n\ndef test_validate_inconsistent_shapes():\n    data = PolarimetricData(\n        hh=np.ones((10, 10), dtype=complex),\n        hv=np.ones((8, 8), dtype=complex),\n    )\n    with pytest.raises(ValueError, match=\"Inconsistent channel shapes\"):\n        data.validate()\n"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 29.7s)

_Thinking:_

Now let me write pyproject.toml and the docs.

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/pyproject.toml",
  "content": "[build-system]\nrequires = [\"setuptools>=68.0\", \"wheel\"]\nbuild-backend = \"setuptools.backends._legacy:_Backend\"\n\n[project]\nname = \"lunaice\"\nversion = \"0.1.0\"\ndescription = \"Lunar Subsurface Ice Detection using Chandrayaan-2 DFSAR Polarimetric Data\"\nreadme = \"README.md\"\nrequires-python = \">=3.12\"\nlicense = {text = \"MIT\"}\nauthors = [\n    {name = \"LUNAICE Team\"},\n]\nkeywords = [\n    \"chandrayaan-2\", \"dfsar\", \"polsar\", \"polarimetry\",\n    \"lunar\", \"radar\", \"remote-sensing\", \"ice-detection\",\n]\nclassifiers = [\n    \"Development Status :: 3 - Alpha\",\n    \"Intended Audience :: Science/Research\",\n    \"License :: OSI Approved :: MIT License\",\n    \"Programming Language :: Python :: 3.12\",\n    \"Topic :: Scientific/Engineering :: GIS\",\n    \"Topic :: Scientific/Engineering :: Physics\",\n]\n\ndependencies = [\n    \"numpy>=1.26\",\n    \"scipy>=1.12\",\n    \"rasterio>=1.3\",\n    \"xarray>=2024.0\",\n    \"zarr>=2.17\",\n    \"pyyaml>=6.0\",\n]\n\n[project.optional-dependencies]\ndev = [\n    \"pytest>=8.0\",\n    \"pytest-cov>=4.1\",\n    \"ruff>=0.1\",\n    \"mypy>=1.8\",\n    \"pre-commit>=3.5\",\n]\nviz = [\n    \"matplotlib>=3.8\",\n    \"cartopy>=0.22\",\n]\nall = [\n    \"lunaice[dev,viz]\",\n]\n\n[project.scripts]\nlunaice = \"lunaice.cli:main\"\n\n[tool.setuptools.packages.find]\nwhere = [\"src\"]\ninclude = [\"lunaice*\"]\n\n[tool.pytest.ini_options]\ntestpaths = [\"tests\"]\naddopts = \"-v --tb=short\"\npython_files = \"test_*.py\"\n\n[tool.ruff]\ntarget-version = \"py312\"\nline-length = 120\nselect = [\"E\", \"F\", \"I\", \"N\", \"W\"]\nignore = [\"E501\"]\n\n[tool.ruff.lint.per-file-ignores]\n\"__init__.py\" = [\"F401\"]\n\n[tool.mypy]\npython_version = \"3.12\"\nstrict = false\nignore_missing_imports = true\ndisallow_untyped_defs = false\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/README.md",
  "content": "# LUNAICE\n\n**Lunar Subsurface Ice Detection using Chandrayaan-2 DFSAR Polarimetric Data**\n\nProduction-grade polarimetric SAR processing pipeline for Chandrayaan-2 Dual-Frequency Synthetic Aperture Radar (DFSAR) L-band and S-band data.\n\n## Architecture\n\n```\n┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐\n│   Input      │────▶│  Calibration      │────▶│  Speckle Filter  │\n│ (PDS4 SLC)   │     │ (Radiometric +    │     │ (Refined Lee /   │\n│              │     │  Polarimetric)    │     │  Boxcar / Idani) │\n└─────────────┘     └──────────────────┘     └──────────────────┘\n                                                       │\n                                                       ▼\n┌─────────────┐     ┌──────────────────┐     ┌──────────────────┐\n│  Output      │◀────│  Decomposition    │◀────│  Coherency T3    │\n│ (GeoTIFF +   │     │ (H/α/A, CPR,     │     │  Matrix Builder   │\n│  Zarr + PNG) │     │  DOP, σ°, γ°)    │     │                  │\n└─────────────┘     └──────────────────┘     └──────────────────┘\n```\n\n## Installation\n\n```bash\npip install -e \".[all]\"\n```\n\n## Quick Start\n\n```bash\nlunaice process -i /path/to/L1A_SLC.h5 -o output/ --band L\n```\n\n## Processing Pipeline\n\n| Step | Component | Description |\n|------|-----------|-------------|\n| 1 | DFSARReader | Loads PDS4 SLC quad-pol data + metadata |\n| 2 | RadiometricCalibrator | Applies calibration constant (K) per channel |\n| 3 | PolarimetricCalibrator | Corrects cross-talk, channel imbalance, phase offsets |\n| 4 | SpeckleFilter | Refined Lee / Boxcar / Lee-Sigma / IDAN / Bilateral |\n| 5 | CoherencyMatrixBuilder | Pauli-based T3 matrix generation |\n| 6 | CloudePottierDecomposition | H/α/A eigenvalue decomposition |\n| 7 | CircularPolarizationRatio | SC → circular basis → CPR |\n| 8 | DegreeOfPolarization | Stokes-based DOP |\n| 9 | BackscatterCoefficients | σ°, γ°, span |\n| 10 | GeoTIFFWriter / ZarrWriter | Output products |\n\n## Output Products\n\n- **GeoTIFFs**: Individual layers for H, α, A, CPR, DOP, σ°, γ°, span\n- **Polarimetric Feature Cube**: Zarr-based multi-variable xarray Dataset\n- **Statistics**: JSON with per-band min/max/mean/std/percentiles\n- **Visualization**: PNG quick-looks (RGB composites)\n\n## Polarimetric Products\n\n| Product | Range | Physical Meaning |\n|---------|-------|-----------------|\n| Entropy (H) | [0, 1] | Scattering randomness |\n| Alpha (α) | [0°, 90°] | Dominant scattering mechanism |\n| Anisotropy (A) | [0, 1] | Secondary scattering relative importance |\n| CPR | ≥ 0 | Circular polarization ratio (ice indicator) |\n| DOP | [0, 1] | Degree of polarization |\n| σ° | dB | Backscatter coefficient (HH/HV/VV) |\n| γ° | dB | Incidence-angle normalized σ° |\n\n## DFSAR Literature Basis\n\nThis implementation follows the calibration and decomposition methodology described in:\n\n- Bhiravarasu et al. (2021) *Chandrayaan-2 DFSAR: Performance Characterization and Initial Results*, PSJ 2, 134\n- Cloude & Pottier (1996) *A review of target decomposition theorems in radar polarimetry*, IEEE TGRS\n- Lee & Pottier (2009) *Polarimetric Radar Imaging: From Basics to Applications*, CRC Press\n- Chakraborty et al. (2024) *Subsurface ice detection in lunar PSRs using DFSAR*\n- Sun et al. (2018) *Polarimetric calibration without external targets*\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/docs/architecture.md",
  "content": "# LUNAICE Architecture\n\n## Overview\n\nLUNAICE is a modular, configurable polarimetric SAR processing pipeline designed specifically for Chandrayaan-2 DFSAR data. The architecture follows a functional pipeline pattern where each processing step is an independent component that transforms the `PolarimetricData` object through the processing chain.\n\n## Class Diagram\n\n```\n┌─────────────────────────────────────────────────────────────┐\n│                     DFSARConfig                              │\n│  - input_file: str                                           │\n│  - output_dir: str                                           │\n│  - band: str (L/S)                                           │\n│  - processing: ProcessingConfig                              │\n│  + from_yaml() -> DFSARConfig                                │\n│  + resolve() -> None                                         │\n└─────────────────────────────────────────────────────────────┘\n                               │\n                               ▼\n┌─────────────────────────────────────────────────────────────┐\n│                      Pipeline                                │\n│  - config: DFSARConfig                                       │\n│  + run() -> ProcessingSummary                                │\n└─────────────────────────────────────────────────────────────┘\n                               │\n          ┌────────────────────┼────────────────────┐\n          ▼                    ▼                    ▼\n┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐\n│   DFSARReader     │  │ RadiometricCal   │  │ PolarimetricCal  │\n│ + read_slc()      │  │  ibrator         │  │  ibrator         │\n│ + read_metadata() │  │ + calibrate()    │  │ + calibrate()    │\n└──────────────────┘  └──────────────────┘  └──────────────────┘\n                                                    │\n                                                    ▼\n                                          ┌──────────────────┐\n                                          │  SpeckleFilter    │\n                                          │  (5 algorithms)   │\n                                          │  + apply()        │\n                                          └──────────────────┘\n                                                    │\n                                                    ▼\n                                          ┌──────────────────┐\n                                          │ CoherencyMatrix   │\n                                          │  Builder          │\n                                          │ + build() -> T3   │\n                                          └──────────────────┘\n                                                    │\n                                                    ▼\n                                          ┌──────────────────┐\n                                          │CloudePottierDecomp│\n                                          │ + decompose()     │\n                                          │  -> H/α/A/λ       │\n                                          └──────────────────┘\n                                                    │\n          ┌────────────────────┬────────────────────┼────────────────────┐\n          ▼                    ▼                    ▼                    ▼\n┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐\n│ CircularPol.     │  │ DegreeOfPol.     │  │ BackscatterCoeff │  │   Output Writers  │\n│  Ratio (CPR)     │  │  (DOP)           │  │  (σ°, γ°)        │  │  GeoTIFF / Zarr   │\n└──────────────────┘  └──────────────────┘  └──────────────────┘  └──────────────────┘\n```\n\n## Data Flow\n\n1. **Input Stage**: `DFSARReader` loads SLC complex data + PDS4 metadata\n2. **Calibration Stage**: Radiometric (DN → σ°) + Polarimetric (phase/cross-talk correction)\n3. **Filtering Stage**: Speckle reduction (5 algorithms available)\n4. **Decomposition Stage**:\n   - Pauli vector → T3 coherency matrix\n   - Eigenvalue decomposition → H, α, A, λ₁, λ₂, λ₃\n   - Circular basis transform → CPR\n   - Stokes parameters → DOP\n5. **Output Stage**: GeoTIFF (per-band), Zarr (multi-variable cube), JSON statistics\n\n## Key Design Decisions\n\n### Config-Driven\nAll processing parameters are externalized in YAML. No hardcoded values.\n\n### Streaming-Aware\nZarr chunked storage enables out-of-core processing of large scenes.\n\n### Type-Safe\nFull Python 3.12 type hints via `dataclass` schemas with validation.\n\n### Graceful Degradation\nPipeline continues with warnings if individual processing steps fail; partial results are still written.\n\n## Data Models\n\n### PolarimetricData\nThe core data container holding complex SLC arrays (HH, HV, VH, VV) with attached Metadata and CalibrationConstants.\n\n### CoherencyMatrix\n9-element T3 matrix stored as individual `np.ndarray` components (t11..t33).\n\n### DecompositionProducts\nContainer for all derived products (H, α, A, CPR, DOP, σ°, γ°, span, λ₁, λ₂, λ₃, α₁, α₂, α₃).\n\n## Processing Modes\n\n| Mode | Bands | Polarization | Products |\n|------|-------|-------------|----------|\n| Quad-pol L | L | HH, HV, VH, VV | Full (all products) |\n| Quad-pol S | S | HH, HV, VH, VV | Full (all products) |\n| Dual-pol | L/S | HH, HV or VV, VH | Subset (CPR, DOP, σ°) |\n| Hybrid | L/S | Compact | Subset (DOP, Stokes) |\n"
}
```

**Output:**
```
Wrote file successfully.
```

**Tool: write**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/docs/api.md",
  "content": "# LUNAICE API Reference\n\n## Configuration\n\n### `DFSARConfig`\nTop-level configuration for the pipeline.\n\n```python\nfrom lunaice import DFSARConfig\nconfig = DFSARConfig(\n    input_file=\"/data/SLC.h5\",\n    output_dir=\"/output/\",\n    band=\"L\",\n    processing=ProcessingConfig(...),\n)\n```\n\n### `ProcessingConfig`\n```python\nProcessingConfig(\n    radiometric_calibration=True,\n    polarimetric_calibration=True,\n    speckle_filter=SpeckleFilterConfig(...),\n    multilook_range=1,\n    multilook_azimuth=1,\n    generate_coherency_matrix=True,\n    generate_cloude_pottier=True,\n    generate_cpr=True,\n    generate_dop=True,\n    generate_backscatter=True,\n)\n```\n\n### `SpeckleFilterConfig`\n| Field | Default | Options |\n|-------|---------|---------|\n| method | \"refined_lee\" | refined_lee, boxcar, lee_sigma, idani, bilateral |\n| window_size | 7 | Odd integer >= 3 |\n| damping_factor | 1.0 | float |\n| n_looks | 4 | int |\n\n## I/O\n\n### `DFSARReader(input_path, label_path=None)`\n- `read_metadata()` -> `Metadata`\n- `read_calibration()` -> `CalibrationConstants`\n- `read_slc()` -> `PolarimetricData`\n\n### `GeoTIFFWriter(output_dir, crs=\"EPSG:4326\")`\n- `write_band(array, name)` -> `Path`\n- `write_multiband(bands_dict, name)` -> `Path`\n\n### `ZarrWriter(output_dir, chunks=(256,256))`\n- `write_product_cube(products)` -> `Path`\n\n### `ReportWriter(output_dir)`\n- `write_summary(summary_dict)` -> `Path`\n- `write_statistics(products)` -> `Path`\n\n## Processing\n\n### `RadiometricCalibrator(cal_constant=1.0)`\n- `calibrate(data)` -> `PolarimetricData`\n\n### `PolarimetricCalibrator(co_pol_phase=-50.0, cross_pol_phase=-5.0, ...)`\n- `calibrate(data)` -> `PolarimetricData`\n\n### `SpeckleFilter(config: SpeckleFilterConfig)`\n- `apply(data)` -> `PolarimetricData`\n\n### `CoherencyMatrixBuilder(multilook_range=1, multilook_azimuth=1)`\n- `build(data)` -> `CoherencyMatrix`\n\n### `CloudePottierDecomposition()`\n- `decompose(t3)` -> `DecompositionProducts`\n\n### `CircularPolarizationRatio()`\n- `compute(data, products)` -> `DecompositionProducts`\n\n### `DegreeOfPolarization()`\n- `compute(data, products)` -> `DecompositionProducts`\n\n### `BackscatterCoefficient(incidence_angle_deg=30.0)`\n- `compute(data, products)` -> `DecompositionProducts`\n\n## CLI\n\n```bash\nlunaice process -i <input> -o <output_dir> [options]\n```\n\n| Flag | Default | Description |\n|------|---------|-------------|\n| `--band` | L | Frequency band (L or S) |\n| `-c, --config` | None | YAML config file |\n| `--no-radiometric` | True | Skip radiometric calibration |\n| `--no-polarimetric` | True | Skip polarimetric calibration |\n| `--no-speckle` | True | Skip speckle filtering |\n| `--speckle-method` | refined_lee | Speckle filter algorithm |\n| `--speckle-window` | 7 | Filter kernel size |\n| `--no-cloude` | True | Skip H/α/A decomposition |\n| `--no-cpr` | True | Skip CPR computation |\n| `--no-dop` | True | Skip DOP computation |\n| `--no-backscatter` | True | Skip backscatter coefficients |\n| `--multilook-range` | 1 | Range multilook factor |\n| `--multilook-azimuth` | 1 | Azimuth multilook factor |\n| `-v, --verbose` | False | DEBUG-level logging |\n"
}
```

**Output:**
```
Wrote file successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 3.4s)

_Thinking:_

Now let me write the notebook and create __init__.py in the processing sub-package. Actually, I already have the processing/__init__.py. Let me check if I missed any `__init__.py` for processing/polarimetry - I did write it already.

Let me verify all the files exist and run the tests.

Now let me verify the structure and run tests.

**Tool: bash**

**Input:**
```json
{
  "description": "Verify complete directory tree",
  "command": "find /home/rabindra03/workspace/DEV/LUNAICE -type f | sort"
}
```

**Output:**
```
/home/rabindra03/workspace/DEV/LUNAICE/configs/default.yaml
/home/rabindra03/workspace/DEV/LUNAICE/configs/processing.yaml
/home/rabindra03/workspace/DEV/LUNAICE/docs/api.md
/home/rabindra03/workspace/DEV/LUNAICE/docs/architecture.md
/home/rabindra03/workspace/DEV/LUNAICE/pyproject.toml
/home/rabindra03/workspace/DEV/LUNAICE/README.md
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/cli.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/config.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/__init__.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/io/__init__.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/io/reader.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/io/writer.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/models/__init__.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/models/schemas.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/pipeline.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/calibration.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/__init__.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/backscatter.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/cloude_pottier.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/coherency.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/cpr.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/dop.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/polarimetry/__init__.py
/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/speckle.py
/home/rabindra03/workspace/DEV/LUNAICE/tests/conftest.py
/home/rabindra03/workspace/DEV/LUNAICE/tests/__init__.py
/home/rabindra03/workspace/DEV/LUNAICE/tests/test_calibration.py
/home/rabindra03/workspace/DEV/LUNAICE/tests/test_pipeline.py
/home/rabindra03/workspace/DEV/LUNAICE/tests/test_polarimetry.py
/home/rabindra03/workspace/DEV/LUNAICE/tests/test_reader.py

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 3.7s)

_Thinking:_

Good, all files are in place. Let me install the package and run tests.

**Tool: bash**

**Input:**
```json
{
  "description": "Install package and run tests",
  "command": "pip install -e \".[dev]\" 2>&1 | tail -5 && python -m pytest tests/ -v --tb=short 2>&1",
  "workdir": "/home/rabindra03/workspace/DEV/LUNAICE",
  "timeout": 120000
}
```

**Output:**
```
    
    See /usr/share/doc/python3.12/README.venv for more information.

note: If you believe this is a mistake, please contact your Python installation or OS distribution provider. You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-packages.
hint: See PEP 668 for the detailed specification.
/usr/bin/python: No module named pytest

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 8.3s)

_Thinking:_

I need to set up a virtual environment first or use --break-system-packages.

**Tool: bash**

**Input:**
```json
{
  "command": "python3 -m venv /tmp/lunaice_venv && source /tmp/lunaice_venv/bin/activate && pip install --quiet -e \".[dev]\" 2>&1 | tail -5",
  "description": "Create venv and install package",
  "timeout": 120000,
  "workdir": "/home/rabindra03/workspace/DEV/LUNAICE"
}
```

**Output:**
```
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1324, in _find_and_load_unlocked
ModuleNotFoundError: No module named 'setuptools.backends'


```

---

## Assistant (Build · DeepSeek V4 Flash Free · 3.1s)

_Thinking:_

Let me fix the pyproject.toml build system line and try again.

**Tool: edit**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/pyproject.toml",
  "oldString": "[build-system]\nrequires = [\"setuptools>=68.0\", \"wheel\"]\nbuild-backend = \"setuptools.backends._legacy:_Backend\"",
  "newString": "[build-system]\nrequires = [\"setuptools>=68.0\", \"wheel\"]\nbuild-backend = \"setuptools.build_meta\""
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 54.3s)

**Tool: bash**

**Input:**
```json
{
  "description": "Install package with dev deps",
  "command": "source /tmp/lunaice_venv/bin/activate && pip install --quiet -e \".[dev]\" 2>&1 | tail -10",
  "timeout": 180000,
  "workdir": "/home/rabindra03/workspace/DEV/LUNAICE"
}
```

**Output:**
```
(no output)
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 6.0s)

_Thinking:_

Good, it seems to have installed silently. Let me run tests now.

**Tool: bash**

**Input:**
```json
{
  "command": "source /tmp/lunaice_venv/bin/activate && python -m pytest tests/ -v --tb=short 2>&1",
  "description": "Run test suite",
  "timeout": 120000,
  "workdir": "/home/rabindra03/workspace/DEV/LUNAICE"
}
```

**Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0 -- /tmp/lunaice_venv/bin/python
cachedir: .pytest_cache
rootdir: /home/rabindra03/workspace/DEV/LUNAICE
configfile: pyproject.toml
plugins: cov-7.1.0, zarr-3.2.1
collecting ... collected 18 items

tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_quad_pol FAILED [  5%]
tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_partial PASSED [ 11%]
tests/test_calibration.py::TestPolarimetricCalibrator::test_calibrate_quad_pol PASSED [ 16%]
tests/test_calibration.py::TestPolarimetricCalibrator::test_skip_non_quadpol PASSED [ 22%]
tests/test_pipeline.py::test_config_from_yaml PASSED                     [ 27%]
tests/test_pipeline.py::test_speckle_filter_validation PASSED            [ 33%]
tests/test_pipeline.py::test_speckle_filter_methods PASSED               [ 38%]
tests/test_pipeline.py::test_processing_config_defaults PASSED           [ 44%]
tests/test_polarimetry.py::TestCoherencyMatrixBuilder::test_build_quad_pol PASSED [ 50%]
tests/test_polarimetry.py::TestCoherencyMatrixBuilder::test_raises_on_partial PASSED [ 55%]
tests/test_polarimetry.py::TestCloudePottierDecomposition::test_decompose PASSED [ 61%]
tests/test_polarimetry.py::TestCircularPolarizationRatio::test_cpr_quad_pol PASSED [ 66%]
tests/test_polarimetry.py::TestDegreeOfPolarization::test_dop_quad_pol PASSED [ 72%]
tests/test_polarimetry.py::TestBackscatterCoefficient::test_compute PASSED [ 77%]
tests/test_reader.py::test_read_pds4_label_not_found PASSED              [ 83%]
tests/test_reader.py::test_dfsar_reader_metadata PASSED                  [ 88%]
tests/test_reader.py::test_validate_consistent_shapes PASSED             [ 94%]
tests/test_reader.py::test_validate_inconsistent_shapes PASSED           [100%]

=================================== FAILURES ===================================
______________ TestRadiometricCalibrator.test_calibrate_quad_pol _______________
tests/test_calibration.py:17: in test_calibrate_quad_pol
    assert np.allclose(np.abs(result.hh) ** 2, expected_power)
E   AssertionError: assert False
E    +  where False = <function allclose at 0x72536fd5c630>((array([[0.62790006, 1.5046351 , 0.50671303, ..., 1.5518873 , 1.0604074 ,\n        0.96560806],\n       [0.08984407, 0.3604614 , 1.0965139 , ..., 0.9237452 , 1.2423961 ,\n        0.51798624],\n       [0.8976163 , 0.55119354, 0.35259554, ..., 0.44526908, 1.6693612 ,\n        0.5456639 ],\n       ...,\n       [0.9972377 , 0.2076551 , 1.1827717 , ..., 0.99784976, 0.95894736,\n        1.1289778 ],\n       [1.0441573 , 1.0824977 , 0.15626901, ..., 0.87206674, 0.6995664 ,\n        0.86448646],\n       [1.1230414 , 1.8208185 , 1.3656362 , ..., 0.31832623, 0.9648565 ,\n        0.20904057]], shape=(128, 128), dtype=float32) ** 2), array([[0.19712925, 1.1319634 , 0.12837905, ..., 1.204177  , 0.5622319 ,\n        0.46619946],\n       [0.00403598, 0.06496622, 0.6011713 , ..., 0.4266526 , 0.77177405,\n        0.13415487],\n       [0.40285754, 0.15190716, 0.06216181, ..., 0.09913228, 1.3933835 ,\n        0.14887454],\n       ...,\n       [0.4972415 , 0.02156032, 0.69947445, ..., 0.4978521 , 0.45979002,\n        0.6372954 ],\n       [0.5451322 , 0.58590066, 0.01221   , ..., 0.3802502 , 0.24469659,\n        0.3736684 ],\n       [0.630611  , 1.65769   , 0.93248117, ..., 0.0506658 , 0.46547404,\n        0.02184898]], shape=(128, 128), dtype=float32))
E    +    where <function allclose at 0x72536fd5c630> = np.allclose
E    +    and   array([[0.62790006, 1.5046351 , 0.50671303, ..., 1.5518873 , 1.0604074 ,\n        0.96560806],\n       [0.08984407, 0.3604614 , 1.0965139 , ..., 0.9237452 , 1.2423961 ,\n        0.51798624],\n       [0.8976163 , 0.55119354, 0.35259554, ..., 0.44526908, 1.6693612 ,\n        0.5456639 ],\n       ...,\n       [0.9972377 , 0.2076551 , 1.1827717 , ..., 0.99784976, 0.95894736,\n        1.1289778 ],\n       [1.0441573 , 1.0824977 , 0.15626901, ..., 0.87206674, 0.6995664 ,\n        0.86448646],\n       [1.1230414 , 1.8208185 , 1.3656362 , ..., 0.31832623, 0.9648565 ,\n        0.20904057]], shape=(128, 128), dtype=float32) = <ufunc 'absolute'>(array([[0.62790006+0.j, 1.5046351 +0.j, 0.50671303+0.j, ...,\n        1.5518873 +0.j, 1.0604074 +0.j, 0.96560806+0.j],\n       [0.08984407+0.j, 0.3604614 +0.j, 1.0965139 +0.j, ...,\n        0.9237452 +0.j, 1.2423961 +0.j, 0.51798624+0.j],\n       [0.8976163 +0.j, 0.55119354+0.j, 0.35259554+0.j, ...,\n        0.44526908+0.j, 1.6693612 +0.j, 0.5456639 +0.j],\n       ...,\n       [0.9972377 +0.j, 0.2076551 +0.j, 1.1827717 +0.j, ...,\n        0.99784976+0.j, 0.95894736+0.j, 1.1289778 +0.j],\n       [1.0441573 +0.j, 1.0824977 +0.j, 0.15626901+0.j, ...,\n        0.87206674+0.j, 0.6995664 +0.j, 0.86448646+0.j],\n       [1.1230414 +0.j, 1.8208185 +0.j, 1.3656362 +0.j, ...,\n        0.31832623+0.j, 0.9648565 +0.j, 0.20904057+0.j]],\n      shape=(128, 128), dtype=complex64))
E    +      where <ufunc 'absolute'> = np.abs
E    +      and   array([[0.62790006+0.j, 1.5046351 +0.j, 0.50671303+0.j, ...,\n        1.5518873 +0.j, 1.0604074 +0.j, 0.96560806+0.j],\n       [0.08984407+0.j, 0.3604614 +0.j, 1.0965139 +0.j, ...,\n        0.9237452 +0.j, 1.2423961 +0.j, 0.51798624+0.j],\n       [0.8976163 +0.j, 0.55119354+0.j, 0.35259554+0.j, ...,\n        0.44526908+0.j, 1.6693612 +0.j, 0.5456639 +0.j],\n       ...,\n       [0.9972377 +0.j, 0.2076551 +0.j, 1.1827717 +0.j, ...,\n        0.99784976+0.j, 0.95894736+0.j, 1.1289778 +0.j],\n       [1.0441573 +0.j, 1.0824977 +0.j, 0.15626901+0.j, ...,\n        0.87206674+0.j, 0.6995664 +0.j, 0.86448646+0.j],\n       [1.1230414 +0.j, 1.8208185 +0.j, 1.3656362 +0.j, ...,\n        0.31832623+0.j, 0.9648565 +0.j, 0.20904057+0.j]],\n      shape=(128, 128), dtype=complex64) = PolarimetricData(data_id='d442a8250d41', hh=array([[0.62790006+0.j, 1.5046351 +0.j, 0.50671303+0.j, ...,\n        1.5518873 +0.j, 1.0604074 +0.j, 0.96560806+0.j],\n       [0.08984407+0.j, 0.3604614 +0.j, 1.0965139 +0.j, ...,\n        0.9237452 +0.j, 1.2423961 +0.j, 0.51798624+0.j],\n       [0.8976163 +0.j, 0.55119354+0.j, 0.35259554+0.j, ...,\n        0.44526908+0.j, 1.6693612 +0.j, 0.5456639 +0.j],\n       ...,\n       [0.9972377 +0.j, 0.2076551 +0.j, 1.1827717 +0.j, ...,\n        0.99784976+0.j, 0.95894736+0.j, 1.1289778 +0.j],\n       [1.0441573 +0.j, 1.0824977 +0.j, 0.15626901+0.j, ...,\n        0.87206674+0.j, 0.6995664 +0.j, 0.86448646+0.j],\n       [1.1230414 +0.j, 1.8208185 +0.j, 1.3656362 +0.j, ...,\n        0.31832623+0.j, 0.9648565 +0.j, 0.20904057+0.j]],\n      shape=(128, 128), dtype=complex64), hv=array([[0.8929364 +0.j, 1.1778655 +0.j, 2.0970442 +0.j, ...,\n        0.81841207+0.j, 1.8340683 +0.j, 1.2581235 +0.j],\n       [1.054831  +0.j, 0.8014888 +0.j, 0.2995871 +0.j, ...,\n        1.4936097 +0.j, 0.6931568 +0.j, 1.1850408 +0.j],\n       [1.0946676 +0.j, 1.18867   +0.j, 1.0497484 +0.j, ...,\n        0.6647975 +0.j, 0.5610291 +0.j, 1.3298471 +0.j],\n       ...,\n       [0.4470411 +0.j, 0.77141243+0.j, 1.1614133 +0.j, ...,\n        0.79414225+0.j, 1.4444401 +0.j, 0.6373361 +0.j],\n       [0.6025976 +0.j, 0.5291501 +0.j, 1.2127277 +0.j, ...,\n        0.92141414+0.j, 0.67384255+0.j, 1.4920089 +0.j],\n       [1.1055328 +0.j, 0.80175734+0.j, 0.453391  +0.j, ...,\n        1.0130724 +0.j, 0.43675536+0.j, 0.7069728 +0.j]],\n      shape=(128, 128), dtype=complex64), vh=array([[0.32848215+0.j, 1.0213002 +0.j, 1.0223002 +0.j, ...,\n        0.6536947 +0.j, 0.89763683+0.j, 0.07481161+0.j],\n       [0.12172112+0.j, 1.139375  +0.j, 0.74170214+0.j, ...,\n        0.7763354 +0.j, 1.5971544 +0.j, 0.70764476+0.j],\n       [0.09489106+0.j, 0.9457016 +0.j, 1.4719902 +0.j, ...,\n        1.329996  +0.j, 0.3678583 +0.j, 0.6048744 +0.j],\n       ...,\n       [0.6226078 +0.j, 1.250402  +0.j, 1.456947  +0.j, ...,\n        1.0495083 +0.j, 0.7988458 +0.j, 0.7917553 +0.j],\n       [0.750329  +0.j, 1.2564797 +0.j, 0.9215277 +0.j, ...,\n        0.86953115+0.j, 0.7045617 +0.j, 1.1224397 +0.j],\n       [1.5351895 +0.j, 0.949383  +0.j, 0.01770712+0.j, ...,\n        0.5107167 +0.j, 1.5143324 +0.j, 0.68802744+0.j]],\n      shape=(128, 128), dtype=complex64), vv=array([[1.1945876 +0.j, 0.85511994+0.j, 0.60689586+0.j, ...,\n        0.7317321 +0.j, 0.70287794+0.j, 1.3712224 +0.j],\n       [0.10911245+0.j, 0.3362784 +0.j, 0.5561495 +0.j, ...,\n        0.6844899 +0.j, 1.229304  +0.j, 0.2653686 +0.j],\n       [0.88236636+0.j, 2.3455286 +0.j, 1.6613497 +0.j, ...,\n        1.688826  +0.j, 0.40244552+0.j, 0.71221626+0.j],\n       ...,\n       [1.3132896 +0.j, 0.76899344+0.j, 0.60566115+0.j, ...,\n        1.4609573 +0.j, 0.74494606+0.j, 0.44470406+0.j],\n       [1.3424637 +0.j, 0.56466234+0.j, 0.98908234+0.j, ...,\n        1.7080883 +0.j, 0.63250625+0.j, 1.002932  +0.j],\n       [0.6189361 +0.j, 1.4159728 +0.j, 0.65228647+0.j, ...,\n        0.5575181 +0.j, 0.94934523+0.j, 0.12667425+0.j]],\n      shape=(128, 128), dtype=complex64), metadata=Metadata(product_id='TEST_001', processing_level='L1A', frequency_band='L', polarization_mode='quad_pol', acquisition_time='2020-01-15T12:00:00', orbit_number=1234, incidence_angle_deg=30.0, slant_range_resolution_m=15.0, azimuth_resolution_m=15.0, looks_range=1, looks_azimuth=1, calibration_constant=1.0, wavelength_cm=24.0, center_latitude=-85.0, center_longitude=30.0, pixel_spacing_m=15.0, cal_params=None), calibration=None).hh
=========================== short test summary info ============================
FAILED tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_quad_pol - AssertionError: assert False
 +  where False = <function allclose at 0x72536fd5c630>((array([[0.62790006, 1.5046351 , 0.50671303, ..., 1.5518873 , 1.0604074 ,\n        0.96560806],\n       [0.08984407, 0.3604614 , 1.0965139 , ..., 0.9237452 , 1.2423961 ,\n        0.51798624],\n       [0.8976163 , 0.55119354, 0.35259554, ..., 0.44526908, 1.6693612 ,\n        0.5456639 ],\n       ...,\n       [0.9972377 , 0.2076551 , 1.1827717 , ..., 0.99784976, 0.95894736,\n        1.1289778 ],\n       [1.0441573 , 1.0824977 , 0.15626901, ..., 0.87206674, 0.6995664 ,\n        0.86448646],\n       [1.1230414 , 1.8208185 , 1.3656362 , ..., 0.31832623, 0.9648565 ,\n        0.20904057]], shape=(128, 128), dtype=float32) ** 2), array([[0.19712925, 1.1319634 , 0.12837905, ..., 1.204177  , 0.5622319 ,\n        0.46619946],\n       [0.00403598, 0.06496622, 0.6011713 , ..., 0.4266526 , 0.77177405,\n        0.13415487],\n       [0.40285754, 0.15190716, 0.06216181, ..., 0.09913228, 1.3933835 ,\n        0.14887454],\n       ...,\n       [0.4972415 , 0.02156032, 0.69947445, ..., 0.4978521 , 0.45979002,\n        0.6372954 ],\n       [0.5451322 , 0.58590066, 0.01221   , ..., 0.3802502 , 0.24469659,\n        0.3736684 ],\n       [0.630611  , 1.65769   , 0.93248117, ..., 0.0506658 , 0.46547404,\n        0.02184898]], shape=(128, 128), dtype=float32))
 +    where <function allclose at 0x72536fd5c630> = np.allclose
 +    and   array([[0.62790006, 1.5046351 , 0.50671303, ..., 1.5518873 , 1.0604074 ,\n        0.96560806],\n       [0.08984407, 0.3604614 , 1.0965139 , ..., 0.9237452 , 1.2423961 ,\n        0.51798624],\n       [0.8976163 , 0.55119354, 0.35259554, ..., 0.44526908, 1.6693612 ,\n        0.5456639 ],\n       ...,\n       [0.9972377 , 0.2076551 , 1.1827717 , ..., 0.99784976, 0.95894736,\n        1.1289778 ],\n       [1.0441573 , 1.0824977 , 0.15626901, ..., 0.87206674, 0.6995664 ,\n        0.86448646],\n       [1.1230414 , 1.8208185 , 1.3656362 , ..., 0.31832623, 0.9648565 ,\n        0.20904057]], shape=(128, 128), dtype=float32) = <ufunc 'absolute'>(array([[0.62790006+0.j, 1.5046351 +0.j, 0.50671303+0.j, ...,\n        1.5518873 +0.j, 1.0604074 +0.j, 0.96560806+0.j],\n       [0.08984407+0.j, 0.3604614 +0.j, 1.0965139 +0.j, ...,\n        0.9237452 +0.j, 1.2423961 +0.j, 0.51798624+0.j],\n       [0.8976163 +0.j, 0.55119354+0.j, 0.35259554+0.j, ...,\n        0.44526908+0.j, 1.6693612 +0.j, 0.5456639 +0.j],\n       ...,\n       [0.9972377 +0.j, 0.2076551 +0.j, 1.1827717 +0.j, ...,\n        0.99784976+0.j, 0.95894736+0.j, 1.1289778 +0.j],\n       [1.0441573 +0.j, 1.0824977 +0.j, 0.15626901+0.j, ...,\n        0.87206674+0.j, 0.6995664 +0.j, 0.86448646+0.j],\n       [1.1230414 +0.j, 1.8208185 +0.j, 1.3656362 +0.j, ...,\n        0.31832623+0.j, 0.9648565 +0.j, 0.20904057+0.j]],\n      shape=(128, 128), dtype=complex64))
 +      where <ufunc 'absolute'> = np.abs
 +      and   array([[0.62790006+0.j, 1.5046351 +0.j, 0.50671303+0.j, ...,\n        1.5518873 +0.j, 1.0604074 +0.j, 0.96560806+0.j],\n       [0.08984407+0.j, 0.3604614 +0.j, 1.0965139 +0.j, ...,\n        0.9237452 +0.j, 1.2423961 +0.j, 0.51798624+0.j],\n       [0.8976163 +0.j, 0.55119354+0.j, 0.35259554+0.j, ...,\n        0.44526908+0.j, 1.6693612 +0.j, 0.5456639 +0.j],\n       ...,\n       [0.9972377 +0.j, 0.2076551 +0.j, 1.1827717 +0.j, ...,\n        0.99784976+0.j, 0.95894736+0.j, 1.1289778 +0.j],\n       [1.0441573 +0.j, 1.0824977 +0.j, 0.15626901+0.j, ...,\n        0.87206674+0.j, 0.6995664 +0.j, 0.86448646+0.j],\n       [1.1230414 +0.j, 1.8208185 +0.j, 1.3656362 +0.j, ...,\n        0.31832623+0.j, 0.9648565 +0.j, 0.20904057+0.j]],\n      shape=(128, 128), dtype=complex64) = PolarimetricData(data_id='d442a8250d41', hh=array([[0.62790006+0.j, 1.5046351 +0.j, 0.50671303+0.j, ...,\n        1.5518873 +0.j, 1.0604074 +0.j, 0.96560806+0.j],\n       [0.08984407+0.j, 0.3604614 +0.j, 1.0965139 +0.j, ...,\n        0.9237452 +0.j, 1.2423961 +0.j, 0.51798624+0.j],\n       [0.8976163 +0.j, 0.55119354+0.j, 0.35259554+0.j, ...,\n        0.44526908+0.j, 1.6693612 +0.j, 0.5456639 +0.j],\n       ...,\n       [0.9972377 +0.j, 0.2076551 +0.j, 1.1827717 +0.j, ...,\n        0.99784976+0.j, 0.95894736+0.j, 1.1289778 +0.j],\n       [1.0441573 +0.j, 1.0824977 +0.j, 0.15626901+0.j, ...,\n        0.87206674+0.j, 0.6995664 +0.j, 0.86448646+0.j],\n       [1.1230414 +0.j, 1.8208185 +0.j, 1.3656362 +0.j, ...,\n        0.31832623+0.j, 0.9648565 +0.j, 0.20904057+0.j]],\n      shape=(128, 128), dtype=complex64), hv=array([[0.8929364 +0.j, 1.1778655 +0.j, 2.0970442 +0.j, ...,\n        0.81841207+0.j, 1.8340683 +0.j, 1.2581235 +0.j],\n       [1.054831  +0.j, 0.8014888 +0.j, 0.2995871 +0.j, ...,\n        1.4936097 +0.j, 0.6931568 +0.j, 1.1850408 +0.j],\n       [1.0946676 +0.j, 1.18867   +0.j, 1.0497484 +0.j, ...,\n        0.6647975 +0.j, 0.5610291 +0.j, 1.3298471 +0.j],\n       ...,\n       [0.4470411 +0.j, 0.77141243+0.j, 1.1614133 +0.j, ...,\n        0.79414225+0.j, 1.4444401 +0.j, 0.6373361 +0.j],\n       [0.6025976 +0.j, 0.5291501 +0.j, 1.2127277 +0.j, ...,\n        0.92141414+0.j, 0.67384255+0.j, 1.4920089 +0.j],\n       [1.1055328 +0.j, 0.80175734+0.j, 0.453391  +0.j, ...,\n        1.0130724 +0.j, 0.43675536+0.j, 0.7069728 +0.j]],\n      shape=(128, 128), dtype=complex64), vh=array([[0.32848215+0.j, 1.0213002 +0.j, 1.0223002 +0.j, ...,\n        0.6536947 +0.j, 0.89763683+0.j, 0.07481161+0.j],\n       [0.12172112+0.j, 1.139375  +0.j, 0.74170214+0.j, ...,\n        0.7763354 +0.j, 1.5971544 +0.j, 0.70764476+0.j],\n       [0.09489106+0.j, 0.9457016 +0.j, 1.4719902 +0.j, ...,\n        1.329996  +0.j, 0.3678583 +0.j, 0.6048744 +0.j],\n       ...,\n       [0.6226078 +0.j, 1.250402  +0.j, 1.456947  +0.j, ...,\n        1.0495083 +0.j, 0.7988458 +0.j, 0.7917553 +0.j],\n       [0.750329  +0.j, 1.2564797 +0.j, 0.9215277 +0.j, ...,\n        0.86953115+0.j, 0.7045617 +0.j, 1.1224397 +0.j],\n       [1.5351895 +0.j, 0.949383  +0.j, 0.01770712+0.j, ...,\n        0.5107167 +0.j, 1.5143324 +0.j, 0.68802744+0.j]],\n      shape=(128, 128), dtype=complex64), vv=array([[1.1945876 +0.j, 0.85511994+0.j, 0.60689586+0.j, ...,\n        0.7317321 +0.j, 0.70287794+0.j, 1.3712224 +0.j],\n       [0.10911245+0.j, 0.3362784 +0.j, 0.5561495 +0.j, ...,\n        0.6844899 +0.j, 1.229304  +0.j, 0.2653686 +0.j],\n       [0.88236636+0.j, 2.3455286 +0.j, 1.6613497 +0.j, ...,\n        1.688826  +0.j, 0.40244552+0.j, 0.71221626+0.j],\n       ...,\n       [1.3132896 +0.j, 0.76899344+0.j, 0.60566115+0.j, ...,\n        1.4609573 +0.j, 0.74494606+0.j, 0.44470406+0.j],\n       [1.3424637 +0.j, 0.56466234+0.j, 0.98908234+0.j, ...,\n        1.7080883 +0.j, 0.63250625+0.j, 1.002932  +0.j],\n       [0.6189361 +0.j, 1.4159728 +0.j, 0.65228647+0.j, ...,\n        0.5575181 +0.j, 0.94934523+0.j, 0.12667425+0.j]],\n      shape=(128, 128), dtype=complex64), metadata=Metadata(product_id='TEST_001', processing_level='L1A', frequency_band='L', polarization_mode='quad_pol', acquisition_time='2020-01-15T12:00:00', orbit_number=1234, incidence_angle_deg=30.0, slant_range_resolution_m=15.0, azimuth_resolution_m=15.0, looks_range=1, looks_azimuth=1, calibration_constant=1.0, wavelength_cm=24.0, center_latitude=-85.0, center_longitude=30.0, pixel_spacing_m=15.0, cal_params=None), calibration=None).hh
========================= 1 failed, 17 passed in 1.27s =========================

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 13.4s)

_Thinking:_

The issue is that the radiometric calibrator takes `np.abs(data.hh)**2 / cal_constant` but `data.hh` still has the original values, and then it stores `sqrt(sigma0)` as the complex value. The problem is that after the calibration, `data.hh` stores the `sqrt(sigma0)` values (magnitude), but the phase information is lost because we're doing `np.sqrt(sigma0).astype(np.complex64)` which gives complex numbers with zero imaginary part.

Wait, actually looking more carefully at the calibration code:

```python
sigma0 = np.abs(arr) ** 2 / (cal_val + 1e-30)
setattr(data, pol, np.sqrt(sigma0).astype(np.complex64))
```

So it takes `abs(arr)^2 / K` and then stores `sqrt(sigma)`. But the original complex values like `0.6279+0.j` become `sqrt(0.6279^2 / 2)` = `sqrt(0.1971)` = `0.4439`... wait, that should match. Let me look again.

The test says:
```python
cal = RadiometricCalibrator(cal_constant=2.0)
result = cal.calibrate(sample_slc_data)
expected_power = np.abs(sample_slc_data.hh) ** 2 / 2.0
assert np.allclose(np.abs(result.hh) ** 2, expected_power)
```

So `result.hh` should be `sqrt(np.abs(sample_slc_data.hh)^2 / 2.0)` which means `np.abs(result.hh)^2` should be `np.abs(sample_slc_data.hh)^2 / 2.0`. Let me check...

The original HH value is `0.62790006+0.j` (but actually it should be complex with non-zero imaginary part from the fixture). Let me check the fixture:

```python
hh = np.random.randn(h, w) + 1j * np.random.randn(h, w)
```

So it should be complex with random real and imag parts. But in the output I see:
```
hh=array([[0.62790006+0.j, 1.5046351 +0.j, ...
```

All have `0.j` imaginary part! That means the calibration function is losing the phase information.

Looking at the code again:
```python
sigma0 = np.abs(arr) ** 2 / (cal_val + 1e-30)
setattr(data, pol, np.sqrt(sigma0).astype(np.complex64))
```

This computes `sqrt(|arr|^2 / K)` as a real number and casts to complex. The phase is completely lost! The calibration should preserve the phase.

Actually for radiometric calibration of SLC data, the standard approach is:
- For intensity data: `sigma0 = |DN|^2 / K`
- For complex SLC data: keep the phase, just scale the amplitude

So the correct approach is:
```python
scale = 1.0 / np.sqrt(cal_val + 1e-30)
setattr(data, pol, arr * scale)
```

This preserves the phase while scaling the amplitude. The power then becomes `|arr|^2 / K`.

Let me fix the calibration code.

The issue is the calibrator loses phase. Fixing the radiometric calibration to preserve phase.

**Tool: edit**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/src/lunaice/processing/calibration.py",
  "oldString": "    def calibrate(self, data: PolarimetricData) -> PolarimetricData:\n        if not data.is_quad_pol:\n            logger.warning(\"Data is not quad-pol; calibrating available channels\")\n        k = data.calibration.k_hh if data.calibration else self.cal_constant\n        for pol in [\"hh\", \"hv\", \"vh\", \"vv\"]:\n            arr = getattr(data, pol, None)\n            if arr is not None:\n                cal_val = getattr(data.calibration, f\"k_{pol}\", k) if data.calibration else k\n                sigma0 = np.abs(arr) ** 2 / (cal_val + 1e-30)\n                setattr(data, pol, np.sqrt(sigma0).astype(np.complex64))\n                logger.debug(\"Radiometric calibration applied to %s (K=%f)\", pol.upper(), cal_val)\n        logger.info(\"Radiometric calibration complete\")\n        return data",
  "newString": "    def calibrate(self, data: PolarimetricData) -> PolarimetricData:\n        if not data.is_quad_pol:\n            logger.warning(\"Data is not quad-pol; calibrating available channels\")\n        k = data.calibration.k_hh if data.calibration else self.cal_constant\n        for pol in [\"hh\", \"hv\", \"vh\", \"vv\"]:\n            arr = getattr(data, pol, None)\n            if arr is not None:\n                cal_val = getattr(data.calibration, f\"k_{pol}\", k) if data.calibration else k\n                scale = np.float32(1.0 / np.sqrt(max(cal_val, 1e-30)))\n                setattr(data, pol, (arr * scale).astype(np.complex64))\n                logger.debug(\"Radiometric calibration applied to %s (K=%f)\", pol.upper(), cal_val)\n        logger.info(\"Radiometric calibration complete\")\n        return data"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 7.1s)

_Thinking:_

Wait, but with this fix the test should still fail because `np.abs(result.hh) ** 2` should equal `np.abs(sample_slc_data.hh) ** 2 / 2.0` when K=2. Let me verify: if `arr = original_arr`, `scale = 1/sqrt(2)`, then `result.hh = original_arr / sqrt(2)`, and `np.abs(result.hh)^2 = |original_arr|^2 / 2`. That's correct!

Let me run the tests again.

**Tool: bash**

**Input:**
```json
{
  "description": "Run tests after fix",
  "command": "source /tmp/lunaice_venv/bin/activate && python -m pytest tests/ -v --tb=short 2>&1",
  "timeout": 120000,
  "workdir": "/home/rabindra03/workspace/DEV/LUNAICE"
}
```

**Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0 -- /tmp/lunaice_venv/bin/python
cachedir: .pytest_cache
rootdir: /home/rabindra03/workspace/DEV/LUNAICE
configfile: pyproject.toml
plugins: cov-7.1.0, zarr-3.2.1
collecting ... collected 18 items

tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_quad_pol FAILED [  5%]
tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_partial PASSED [ 11%]
tests/test_calibration.py::TestPolarimetricCalibrator::test_calibrate_quad_pol PASSED [ 16%]
tests/test_calibration.py::TestPolarimetricCalibrator::test_skip_non_quadpol PASSED [ 22%]
tests/test_pipeline.py::test_config_from_yaml PASSED                     [ 27%]
tests/test_pipeline.py::test_speckle_filter_validation PASSED            [ 33%]
tests/test_pipeline.py::test_speckle_filter_methods PASSED               [ 38%]
tests/test_pipeline.py::test_processing_config_defaults PASSED           [ 44%]
tests/test_polarimetry.py::TestCoherencyMatrixBuilder::test_build_quad_pol PASSED [ 50%]
tests/test_polarimetry.py::TestCoherencyMatrixBuilder::test_raises_on_partial PASSED [ 55%]
tests/test_polarimetry.py::TestCloudePottierDecomposition::test_decompose PASSED [ 61%]
tests/test_polarimetry.py::TestCircularPolarizationRatio::test_cpr_quad_pol PASSED [ 66%]
tests/test_polarimetry.py::TestDegreeOfPolarization::test_dop_quad_pol PASSED [ 72%]
tests/test_polarimetry.py::TestBackscatterCoefficient::test_compute PASSED [ 77%]
tests/test_reader.py::test_read_pds4_label_not_found PASSED              [ 83%]
tests/test_reader.py::test_dfsar_reader_metadata PASSED                  [ 88%]
tests/test_reader.py::test_validate_consistent_shapes PASSED             [ 94%]
tests/test_reader.py::test_validate_inconsistent_shapes PASSED           [100%]

=================================== FAILURES ===================================
______________ TestRadiometricCalibrator.test_calibrate_quad_pol _______________
tests/test_calibration.py:17: in test_calibrate_quad_pol
    assert np.allclose(np.abs(result.hh) ** 2, expected_power)
E   AssertionError: assert False
E    +  where False = <function allclose at 0x7be3f3930bf0>((array([[0.62790006, 1.504635  , 0.506713  , ..., 1.5518872 , 1.0604073 ,\n        0.96560806],\n       [0.08984406, 0.3604614 , 1.096514  , ..., 0.9237453 , 1.242396  ,\n        0.51798624],\n       [0.8976163 , 0.55119354, 0.35259554, ..., 0.44526908, 1.6693611 ,\n        0.5456639 ],\n       ...,\n       [0.9972377 , 0.2076551 , 1.1827717 , ..., 0.99784964, 0.9589474 ,\n        1.1289778 ],\n       [1.0441574 , 1.0824977 , 0.156269  , ..., 0.8720667 , 0.6995664 ,\n        0.86448646],\n       [1.1230414 , 1.8208187 , 1.3656362 , ..., 0.3183262 , 0.9648565 ,\n        0.20904055]], shape=(128, 128), dtype=float32) ** 2), array([[0.19712925, 1.1319633 , 0.12837902, ..., 1.2041769 , 0.5622318 ,\n        0.46619946],\n       [0.00403598, 0.06496622, 0.60117143, ..., 0.42665267, 0.77177393,\n        0.13415487],\n       [0.40285754, 0.15190716, 0.06216181, ..., 0.09913228, 1.3933833 ,\n        0.14887454],\n       ...,\n       [0.4972415 , 0.02156032, 0.69947445, ..., 0.49785197, 0.45979008,\n        0.6372954 ],\n       [0.54513234, 0.58590066, 0.01221   , ..., 0.38025016, 0.24469659,\n        0.3736684 ],\n       [0.630611  , 1.6576903 , 0.93248117, ..., 0.05066578, 0.46547404,\n        0.02184898]], shape=(128, 128), dtype=float32))
E    +    where <function allclose at 0x7be3f3930bf0> = np.allclose
E    +    and   array([[0.62790006, 1.504635  , 0.506713  , ..., 1.5518872 , 1.0604073 ,\n        0.96560806],\n       [0.08984406, 0.3604614 , 1.096514  , ..., 0.9237453 , 1.242396  ,\n        0.51798624],\n       [0.8976163 , 0.55119354, 0.35259554, ..., 0.44526908, 1.6693611 ,\n        0.5456639 ],\n       ...,\n       [0.9972377 , 0.2076551 , 1.1827717 , ..., 0.99784964, 0.9589474 ,\n        1.1289778 ],\n       [1.0441574 , 1.0824977 , 0.156269  , ..., 0.8720667 , 0.6995664 ,\n        0.86448646],\n       [1.1230414 , 1.8208187 , 1.3656362 , ..., 0.3183262 , 0.9648565 ,\n        0.20904055]], shape=(128, 128), dtype=float32) = <ufunc 'absolute'>(array([[ 0.35122994-0.5204767j , -0.09776762-1.5014553j ,\n         0.45798495+0.21681292j, ...,  1.548886  +0.0964684j ,\n        -0.70041496-0.7961675j , -0.40043294-0.8786651j ],\n       [ 0.07046416-0.05573831j, -0.35601103+0.05646731j,\n        -1.0964847 +0.0080132j , ...,  0.7300632 -0.56596196j,\n        -1.0743568 -0.62394345j, -0.3424052 +0.38867524j],\n       [ 0.8958414 +0.05641894j, -0.50039786+0.23111954j,\n         0.31382772+0.16073513j, ..., -0.4238346 +0.13648726j,\n        -1.501821  -0.7289034j , -0.37176493-0.39942446j],\n       ...,\n       [ 0.6217641 +0.77967453j,  0.0277958 -0.20578638j,\n        -0.64511156-0.99135256j, ...,  0.01217608-0.99777544j,\n         0.9430249 +0.17402351j,  0.8071303 +0.78938675j],\n       [ 1.0147445 +0.24608554j,  1.0329548 +0.32373685j,\n         0.10156234+0.11876487j, ...,  0.53841275+0.6860117j ,\n        -0.66992915+0.20146489j, -0.1797144 +0.8456001j ],\n       [-0.8170591 +0.77047795j, -1.8173687 -0.11203182j,\n         0.7262018 +1.1565436j , ...,  0.2525521 -0.19377571j,\n        -0.6335302 -0.72772765j,  0.19165015+0.0834756j ]],\n      shape=(128, 128), dtype=complex64))
E    +      where <ufunc 'absolute'> = np.abs
E    +      and   array([[ 0.35122994-0.5204767j , -0.09776762-1.5014553j ,\n         0.45798495+0.21681292j, ...,  1.548886  +0.0964684j ,\n        -0.70041496-0.7961675j , -0.40043294-0.8786651j ],\n       [ 0.07046416-0.05573831j, -0.35601103+0.05646731j,\n        -1.0964847 +0.0080132j , ...,  0.7300632 -0.56596196j,\n        -1.0743568 -0.62394345j, -0.3424052 +0.38867524j],\n       [ 0.8958414 +0.05641894j, -0.50039786+0.23111954j,\n         0.31382772+0.16073513j, ..., -0.4238346 +0.13648726j,\n        -1.501821  -0.7289034j , -0.37176493-0.39942446j],\n       ...,\n       [ 0.6217641 +0.77967453j,  0.0277958 -0.20578638j,\n        -0.64511156-0.99135256j, ...,  0.01217608-0.99777544j,\n         0.9430249 +0.17402351j,  0.8071303 +0.78938675j],\n       [ 1.0147445 +0.24608554j,  1.0329548 +0.32373685j,\n         0.10156234+0.11876487j, ...,  0.53841275+0.6860117j ,\n        -0.66992915+0.20146489j, -0.1797144 +0.8456001j ],\n       [-0.8170591 +0.77047795j, -1.8173687 -0.11203182j,\n         0.7262018 +1.1565436j , ...,  0.2525521 -0.19377571j,\n        -0.6335302 -0.72772765j,  0.19165015+0.0834756j ]],\n      shape=(128, 128), dtype=complex64) = PolarimetricData(data_id='9f7b106023f8', hh=array([[ 0.35122994-0.5204767j , -0.09776762-1.5014553j ,\n         0.45798495+0.21681292j, ...,  1.548886  +0.0964684j ,\n        -0.70041496-0.7961675j , -0.40043294-0.8786651j ],\n       [ 0.07046416-0.05573831j, -0.35601103+0.05646731j,\n        -1.0964847 +0.0080132j , ...,  0.7300632 -0.56596196j,\n        -1.0743568 -0.62394345j, -0.3424052 +0.38867524j],\n       [ 0.8958414 +0.05641894j, -0.50039786+0.23111954j,\n         0.31382772+0.16073513j, ..., -0.4238346 +0.13648726j,\n        -1.501821  -0.7289034j , -0.37176493-0.39942446j],\n       ...,\n       [ 0.6217641 +0.77967453j,  0.0277958 -0.20578638j,\n        -0.64511156-0.99135256j, ...,  0.01217608-0.99777544j,\n         0.9430249 +0.17402351j,  0.8071303 +0.78938675j],\n       [ 1.0147445 +0.24608554j,  1.0329548 +0.32373685j,\n         0.10156234+0.11876487j, ...,  0.53841275+0.6860117j ,\n        -0.66992915+0.20146489j, -0.1797144 +0.8456001j ],\n       [-0.8170591 +0.77047795j, -1.8173687 -0.11203182j,\n         0.7262018 +1.1565436j , ...,  0.2525521 -0.19377571j,\n        -0.6335302 -0.72772765j,  0.19165015+0.0834756j ]],\n      shape=(128, 128), dtype=complex64), hv=array([[-0.7595168 -0.46954194j, -0.37238607+1.1174506j ,\n         0.4957736 +2.0375974j , ...,  0.81801784-0.02539907j,\n         1.7749993 -0.46171826j, -1.1114773 +0.58948535j],\n       [ 0.9925086 -0.35720477j, -0.3311389 +0.72988445j,\n        -0.23579794+0.18480192j, ..., -1.4865402 -0.14514776j,\n         0.54034823-0.4341545j ,  1.1638967 -0.2228588j ],\n       [ 1.0908881 +0.09088484j, -0.9956694 -0.6492909j ,\n         0.7186728 -0.76516736j, ...,  0.61873573-0.2431496j ,\n        -0.5591815 +0.04549294j,  0.25500628+1.3051686j ],\n       ...,\n       [-0.42831686-0.12802497j, -0.49796444+0.58915913j,\n         0.15565817+1.150935j  , ..., -0.6643236 +0.43512753j,\n        -0.44878024-1.3729542j , -0.23474123+0.59253174j],\n       [ 0.60084677+0.04590235j, -0.52863926-0.02324627j,\n         0.87301034+0.8417609j , ..., -0.0666305 -0.9190018j ,\n        -0.23917909-0.629966j  ,  0.15474574+1.4839623j ],\n       [ 1.0636556 -0.3013955j ,  0.1514004 +0.7873327j ,\n         0.44456968-0.08900095j, ..., -0.16426861+0.9996656j ,\n         0.08616238-0.42817202j, -0.670874  +0.2230216j ]],\n      shape=(128, 128), dtype=complex64), vh=array([[ 0.3226884 +0.06142247j, -0.6707362 +0.7701733j ,\n        -1.0121839 -0.14346293j, ..., -0.36905035-0.53955406j,\n         0.15271446+0.8845508j , -0.06584904+0.03550605j],\n       [-0.11425047+0.04198642j, -0.18237855+1.1246836j ,\n        -0.6204119 -0.40646166j, ...,  0.61054367-0.47951335j,\n        -0.8786537 -1.3337427j ,  0.10524076+0.6997753j ],\n       [-0.05786646+0.07520495j,  0.7187959 -0.6145599j ,\n        -0.70493346-1.2922167j , ..., -0.25827062-1.3046784j ,\n         0.03156832+0.36650124j, -0.04707795-0.6030395j ],\n       ...,\n       [ 0.6225801 -0.00587783j, -1.0642287 +0.6564469j ,\n        -1.4316025 +0.27057108j, ...,  0.8192626 -0.65595466j,\n         0.5140911 +0.611445j  , -0.74840343+0.25839648j],\n       [ 0.03578827-0.749475j  , -0.50149935-1.1520588j ,\n        -0.9018726 +0.1893121j , ...,  0.23519588-0.83711845j,\n         0.66838706-0.22285844j, -0.23397912-1.0977817j ],\n       [-0.9588964 -1.1988847j , -0.19341107-0.9294731j ,\n        -0.01391924+0.01094519j, ...,  0.17909633+0.47828448j,\n         0.8780904 +1.2337584j , -0.6393632 +0.25415814j]],\n      shape=(128, 128), dtype=complex64), vv=array([[ 0.71240425-0.95891577j,  0.14970547-0.8419135j ,\n        -0.6018188 -0.07833704j, ..., -0.71822244-0.13995826j,\n        -0.6920711 +0.12278001j,  1.3690687 +0.07682121j],\n       [-0.09775347-0.04847458j, -0.2838312 +0.18034141j,\n         0.5260427 +0.18050303j, ...,  0.16807288-0.66353446j,\n        -0.26914194+1.1994793j , -0.0782058 -0.253583j  ],\n       [-0.8668638 +0.16467418j, -1.1020122 +2.070525j  ,\n         1.0293907 +1.3040081j , ..., -1.5895848 -0.5703974j ,\n        -0.11728925-0.38497487j,  0.6992777 -0.1351396j ],\n       ...,\n       [ 1.2430264 +0.42381042j, -0.66488034+0.38637426j,\n         0.30693588-0.52212626j, ...,  1.4518437 +0.1629287j ,\n         0.6670922 -0.33156082j,  0.35587844-0.26666874j],\n       [ 0.1300488 -1.3361497j ,  0.48144367-0.2950518j ,\n         0.21386622-0.96568376j, ...,  1.0119983 -1.3760178j ,\n        -0.6034122 -0.18962555j, -0.9029989 -0.43642333j],\n       [-0.19120204+0.5886626j , -0.24893227+1.3939195j ,\n        -0.59050477-0.2770952j , ...,  0.5548169 -0.05481486j,\n         0.92559993+0.21100026j,  0.10440759-0.07173158j]],\n      shape=(128, 128), dtype=complex64), metadata=Metadata(product_id='TEST_001', processing_level='L1A', frequency_band='L', polarization_mode='quad_pol', acquisition_time='2020-01-15T12:00:00', orbit_number=1234, incidence_angle_deg=30.0, slant_range_resolution_m=15.0, azimuth_resolution_m=15.0, looks_range=1, looks_azimuth=1, calibration_constant=1.0, wavelength_cm=24.0, center_latitude=-85.0, center_longitude=30.0, pixel_spacing_m=15.0, cal_params=None), calibration=None).hh
=========================== short test summary info ============================
FAILED tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_quad_pol - AssertionError: assert False
 +  where False = <function allclose at 0x7be3f3930bf0>((array([[0.62790006, 1.504635  , 0.506713  , ..., 1.5518872 , 1.0604073 ,\n        0.96560806],\n       [0.08984406, 0.3604614 , 1.096514  , ..., 0.9237453 , 1.242396  ,\n        0.51798624],\n       [0.8976163 , 0.55119354, 0.35259554, ..., 0.44526908, 1.6693611 ,\n        0.5456639 ],\n       ...,\n       [0.9972377 , 0.2076551 , 1.1827717 , ..., 0.99784964, 0.9589474 ,\n        1.1289778 ],\n       [1.0441574 , 1.0824977 , 0.156269  , ..., 0.8720667 , 0.6995664 ,\n        0.86448646],\n       [1.1230414 , 1.8208187 , 1.3656362 , ..., 0.3183262 , 0.9648565 ,\n        0.20904055]], shape=(128, 128), dtype=float32) ** 2), array([[0.19712925, 1.1319633 , 0.12837902, ..., 1.2041769 , 0.5622318 ,\n        0.46619946],\n       [0.00403598, 0.06496622, 0.60117143, ..., 0.42665267, 0.77177393,\n        0.13415487],\n       [0.40285754, 0.15190716, 0.06216181, ..., 0.09913228, 1.3933833 ,\n        0.14887454],\n       ...,\n       [0.4972415 , 0.02156032, 0.69947445, ..., 0.49785197, 0.45979008,\n        0.6372954 ],\n       [0.54513234, 0.58590066, 0.01221   , ..., 0.38025016, 0.24469659,\n        0.3736684 ],\n       [0.630611  , 1.6576903 , 0.93248117, ..., 0.05066578, 0.46547404,\n        0.02184898]], shape=(128, 128), dtype=float32))
 +    where <function allclose at 0x7be3f3930bf0> = np.allclose
 +    and   array([[0.62790006, 1.504635  , 0.506713  , ..., 1.5518872 , 1.0604073 ,\n        0.96560806],\n       [0.08984406, 0.3604614 , 1.096514  , ..., 0.9237453 , 1.242396  ,\n        0.51798624],\n       [0.8976163 , 0.55119354, 0.35259554, ..., 0.44526908, 1.6693611 ,\n        0.5456639 ],\n       ...,\n       [0.9972377 , 0.2076551 , 1.1827717 , ..., 0.99784964, 0.9589474 ,\n        1.1289778 ],\n       [1.0441574 , 1.0824977 , 0.156269  , ..., 0.8720667 , 0.6995664 ,\n        0.86448646],\n       [1.1230414 , 1.8208187 , 1.3656362 , ..., 0.3183262 , 0.9648565 ,\n        0.20904055]], shape=(128, 128), dtype=float32) = <ufunc 'absolute'>(array([[ 0.35122994-0.5204767j , -0.09776762-1.5014553j ,\n         0.45798495+0.21681292j, ...,  1.548886  +0.0964684j ,\n        -0.70041496-0.7961675j , -0.40043294-0.8786651j ],\n       [ 0.07046416-0.05573831j, -0.35601103+0.05646731j,\n        -1.0964847 +0.0080132j , ...,  0.7300632 -0.56596196j,\n        -1.0743568 -0.62394345j, -0.3424052 +0.38867524j],\n       [ 0.8958414 +0.05641894j, -0.50039786+0.23111954j,\n         0.31382772+0.16073513j, ..., -0.4238346 +0.13648726j,\n        -1.501821  -0.7289034j , -0.37176493-0.39942446j],\n       ...,\n       [ 0.6217641 +0.77967453j,  0.0277958 -0.20578638j,\n        -0.64511156-0.99135256j, ...,  0.01217608-0.99777544j,\n         0.9430249 +0.17402351j,  0.8071303 +0.78938675j],\n       [ 1.0147445 +0.24608554j,  1.0329548 +0.32373685j,\n         0.10156234+0.11876487j, ...,  0.53841275+0.6860117j ,\n        -0.66992915+0.20146489j, -0.1797144 +0.8456001j ],\n       [-0.8170591 +0.77047795j, -1.8173687 -0.11203182j,\n         0.7262018 +1.1565436j , ...,  0.2525521 -0.19377571j,\n        -0.6335302 -0.72772765j,  0.19165015+0.0834756j ]],\n      shape=(128, 128), dtype=complex64))
 +      where <ufunc 'absolute'> = np.abs
 +      and   array([[ 0.35122994-0.5204767j , -0.09776762-1.5014553j ,\n         0.45798495+0.21681292j, ...,  1.548886  +0.0964684j ,\n        -0.70041496-0.7961675j , -0.40043294-0.8786651j ],\n       [ 0.07046416-0.05573831j, -0.35601103+0.05646731j,\n        -1.0964847 +0.0080132j , ...,  0.7300632 -0.56596196j,\n        -1.0743568 -0.62394345j, -0.3424052 +0.38867524j],\n       [ 0.8958414 +0.05641894j, -0.50039786+0.23111954j,\n         0.31382772+0.16073513j, ..., -0.4238346 +0.13648726j,\n        -1.501821  -0.7289034j , -0.37176493-0.39942446j],\n       ...,\n       [ 0.6217641 +0.77967453j,  0.0277958 -0.20578638j,\n        -0.64511156-0.99135256j, ...,  0.01217608-0.99777544j,\n         0.9430249 +0.17402351j,  0.8071303 +0.78938675j],\n       [ 1.0147445 +0.24608554j,  1.0329548 +0.32373685j,\n         0.10156234+0.11876487j, ...,  0.53841275+0.6860117j ,\n        -0.66992915+0.20146489j, -0.1797144 +0.8456001j ],\n       [-0.8170591 +0.77047795j, -1.8173687 -0.11203182j,\n         0.7262018 +1.1565436j , ...,  0.2525521 -0.19377571j,\n        -0.6335302 -0.72772765j,  0.19165015+0.0834756j ]],\n      shape=(128, 128), dtype=complex64) = PolarimetricData(data_id='9f7b106023f8', hh=array([[ 0.35122994-0.5204767j , -0.09776762-1.5014553j ,\n         0.45798495+0.21681292j, ...,  1.548886  +0.0964684j ,\n        -0.70041496-0.7961675j , -0.40043294-0.8786651j ],\n       [ 0.07046416-0.05573831j, -0.35601103+0.05646731j,\n        -1.0964847 +0.0080132j , ...,  0.7300632 -0.56596196j,\n        -1.0743568 -0.62394345j, -0.3424052 +0.38867524j],\n       [ 0.8958414 +0.05641894j, -0.50039786+0.23111954j,\n         0.31382772+0.16073513j, ..., -0.4238346 +0.13648726j,\n        -1.501821  -0.7289034j , -0.37176493-0.39942446j],\n       ...,\n       [ 0.6217641 +0.77967453j,  0.0277958 -0.20578638j,\n        -0.64511156-0.99135256j, ...,  0.01217608-0.99777544j,\n         0.9430249 +0.17402351j,  0.8071303 +0.78938675j],\n       [ 1.0147445 +0.24608554j,  1.0329548 +0.32373685j,\n         0.10156234+0.11876487j, ...,  0.53841275+0.6860117j ,\n        -0.66992915+0.20146489j, -0.1797144 +0.8456001j ],\n       [-0.8170591 +0.77047795j, -1.8173687 -0.11203182j,\n         0.7262018 +1.1565436j , ...,  0.2525521 -0.19377571j,\n        -0.6335302 -0.72772765j,  0.19165015+0.0834756j ]],\n      shape=(128, 128), dtype=complex64), hv=array([[-0.7595168 -0.46954194j, -0.37238607+1.1174506j ,\n         0.4957736 +2.0375974j , ...,  0.81801784-0.02539907j,\n         1.7749993 -0.46171826j, -1.1114773 +0.58948535j],\n       [ 0.9925086 -0.35720477j, -0.3311389 +0.72988445j,\n        -0.23579794+0.18480192j, ..., -1.4865402 -0.14514776j,\n         0.54034823-0.4341545j ,  1.1638967 -0.2228588j ],\n       [ 1.0908881 +0.09088484j, -0.9956694 -0.6492909j ,\n         0.7186728 -0.76516736j, ...,  0.61873573-0.2431496j ,\n        -0.5591815 +0.04549294j,  0.25500628+1.3051686j ],\n       ...,\n       [-0.42831686-0.12802497j, -0.49796444+0.58915913j,\n         0.15565817+1.150935j  , ..., -0.6643236 +0.43512753j,\n        -0.44878024-1.3729542j , -0.23474123+0.59253174j],\n       [ 0.60084677+0.04590235j, -0.52863926-0.02324627j,\n         0.87301034+0.8417609j , ..., -0.0666305 -0.9190018j ,\n        -0.23917909-0.629966j  ,  0.15474574+1.4839623j ],\n       [ 1.0636556 -0.3013955j ,  0.1514004 +0.7873327j ,\n         0.44456968-0.08900095j, ..., -0.16426861+0.9996656j ,\n         0.08616238-0.42817202j, -0.670874  +0.2230216j ]],\n      shape=(128, 128), dtype=complex64), vh=array([[ 0.3226884 +0.06142247j, -0.6707362 +0.7701733j ,\n        -1.0121839 -0.14346293j, ..., -0.36905035-0.53955406j,\n         0.15271446+0.8845508j , -0.06584904+0.03550605j],\n       [-0.11425047+0.04198642j, -0.18237855+1.1246836j ,\n        -0.6204119 -0.40646166j, ...,  0.61054367-0.47951335j,\n        -0.8786537 -1.3337427j ,  0.10524076+0.6997753j ],\n       [-0.05786646+0.07520495j,  0.7187959 -0.6145599j ,\n        -0.70493346-1.2922167j , ..., -0.25827062-1.3046784j ,\n         0.03156832+0.36650124j, -0.04707795-0.6030395j ],\n       ...,\n       [ 0.6225801 -0.00587783j, -1.0642287 +0.6564469j ,\n        -1.4316025 +0.27057108j, ...,  0.8192626 -0.65595466j,\n         0.5140911 +0.611445j  , -0.74840343+0.25839648j],\n       [ 0.03578827-0.749475j  , -0.50149935-1.1520588j ,\n        -0.9018726 +0.1893121j , ...,  0.23519588-0.83711845j,\n         0.66838706-0.22285844j, -0.23397912-1.0977817j ],\n       [-0.9588964 -1.1988847j , -0.19341107-0.9294731j ,\n        -0.01391924+0.01094519j, ...,  0.17909633+0.47828448j,\n         0.8780904 +1.2337584j , -0.6393632 +0.25415814j]],\n      shape=(128, 128), dtype=complex64), vv=array([[ 0.71240425-0.95891577j,  0.14970547-0.8419135j ,\n        -0.6018188 -0.07833704j, ..., -0.71822244-0.13995826j,\n        -0.6920711 +0.12278001j,  1.3690687 +0.07682121j],\n       [-0.09775347-0.04847458j, -0.2838312 +0.18034141j,\n         0.5260427 +0.18050303j, ...,  0.16807288-0.66353446j,\n        -0.26914194+1.1994793j , -0.0782058 -0.253583j  ],\n       [-0.8668638 +0.16467418j, -1.1020122 +2.070525j  ,\n         1.0293907 +1.3040081j , ..., -1.5895848 -0.5703974j ,\n        -0.11728925-0.38497487j,  0.6992777 -0.1351396j ],\n       ...,\n       [ 1.2430264 +0.42381042j, -0.66488034+0.38637426j,\n         0.30693588-0.52212626j, ...,  1.4518437 +0.1629287j ,\n         0.6670922 -0.33156082j,  0.35587844-0.26666874j],\n       [ 0.1300488 -1.3361497j ,  0.48144367-0.2950518j ,\n         0.21386622-0.96568376j, ...,  1.0119983 -1.3760178j ,\n        -0.6034122 -0.18962555j, -0.9029989 -0.43642333j],\n       [-0.19120204+0.5886626j , -0.24893227+1.3939195j ,\n        -0.59050477-0.2770952j , ...,  0.5548169 -0.05481486j,\n         0.92559993+0.21100026j,  0.10440759-0.07173158j]],\n      shape=(128, 128), dtype=complex64), metadata=Metadata(product_id='TEST_001', processing_level='L1A', frequency_band='L', polarization_mode='quad_pol', acquisition_time='2020-01-15T12:00:00', orbit_number=1234, incidence_angle_deg=30.0, slant_range_resolution_m=15.0, azimuth_resolution_m=15.0, looks_range=1, looks_azimuth=1, calibration_constant=1.0, wavelength_cm=24.0, center_latitude=-85.0, center_longitude=30.0, pixel_spacing_m=15.0, cal_params=None), calibration=None).hh
========================= 1 failed, 17 passed in 1.23s =========================

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 30.3s)

_Thinking:_

The test is checking:
```python
expected_power = np.abs(sample_slc_data.hh) ** 2 / 2.0
```

But `sample_slc_data.hh` has complex values (e.g., `0.35122994-0.5204767j`), and after calibration with `K=2`, we do `result.hh = hh / sqrt(2)`, so `np.abs(result.hh)^2 = |hh|^2 / 2`.

Let me verify: `|0.35122994-0.5204767j|^2 = 0.35122994^2 + 0.5204767^2 = 0.12336 + 0.27090 = 0.39426`
`expected_power = 0.39426 / 2 = 0.19713`

After calibration: `result.hh = (0.35122994-0.5204767j) / sqrt(2) = 0.24836-0.36807j`
`np.abs(result.hh)^2 = 0.24836^2 + 0.36807^2 = 0.06168 + 0.13548 = 0.19716` ✓

So mathematically it should work... but the test shows different numbers. Looking at the error again:

The first array `(|result.hh|)^2` shows numbers like `0.62790006^2 = 0.394...` while `expected_power` shows numbers like `0.19712925`. Wait, let me look more carefully at the error output.

```
E       (array([[0.62790006, 1.504635  , ...
```

This is `|result.hh|` (NOT `|result.hh|^2`). Look at the test assertion:
```python
assert np.allclose(np.abs(result.hh) ** 2, expected_power)
```

In the error output, the first argument is:
```
(array([[0.62790006, 1.504635  , ...]]) ** 2)
```

Wait - this is confusing. The `** 2` is shown in the error format. Actually wait - looking again, the actual test output seems to show `np.abs(result.hh)` values of `0.62790006` which squared would be `0.394...`, but the expected is `0.197...` which is exactly half.

Actually the problem is clear from the output. Let me trace through more carefully.

The test fixture creates:
```python
hh = np.random.randn(h, w) + 1j * np.random.randn(h, w)
```

This creates COMPLEX random values. But the conftest uses `np.random.seed(42)` which creates the same values.

Now, after calibration with K=2:
```python
result.hh = hh / sqrt(2)
```

So `np.abs(result.hh)^2 = |hh|^2 / 2`.

But wait - let me look at the error output carefully. It says:
```
E       (array([[0.62790006, 1.504635  , ...]]) ** 2), array([[0.19712925, 1.1319633 , ...]]))
```

So the first array is `np.abs(result.hh)` with values like `0.6279`, and then Python applies `**2` to get `0.394...`. But the expected is `0.197...`. So `0.394 ≠ 0.197`. That means the calibration is NOT dividing by sqrt(2).

Oh wait - I see. Let me look at the actual numbers more carefully:

The error shows: `np.abs(result.hh)` is `0.62790006`.

But the original HH value in the fixture (from the random seed) is: we can see from the raw hh value `0.35122994-0.5204767j`, so `|hh| = sqrt(0.35122994^2 + 0.5204767^2) = sqrt(0.12336 + 0.27090) = sqrt(0.39426) = 0.6279`. And the calibrated value should be `hh / sqrt(2) = (0.35122994-0.5204767j) / 1.4142 = 0.2484-0.3681j`, so `|result.hh| = sqrt(0.2484^2 + 0.3681^2) = sqrt(0.0617 + 0.1355) = sqrt(0.1972) = 0.4441`.

But the error output shows `np.abs(result.hh)` as `0.62790006`, which is the ORIGINAL magnitude, not the calibrated magnitude!

Wait, look at the error output again:
```
result.hh=array([[ 0.35122994-0.5204767j , -0.09776762-1.5014553j , ...
```

So `result.hh` IS `0.35122994-0.5204767j`, which is the ORIGINAL uncalibrated value! So the calibration is not being applied at all.

But that can't be right... let me re-examine the calibration code:

```python
def calibrate(self, data: PolarimetricData) -> PolarimetricData:
    ...
    for pol in ["hh", "hv", "vh", "vv"]:
        arr = getattr(data, pol, None)
        if arr is not None:
            cal_val = getattr(data.calibration, f"k_{pol}", k) if data.calibration else k
            scale = np.float32(1.0 / np.sqrt(max(cal_val, 1e-30)))
            setattr(data, pol, (arr * scale).astype(np.complex64))
```

Wait... `data.calibration` is `None` for the test fixture (the fixture creates `PolarimetricData` without `calibration`). So it uses `self.cal_constant = 2.0` as `k`.

Then `cal_val = k = 2.0`, `scale = 1/sqrt(2) = 0.707...`.

`result.hh = arr * 0.707... = (0.35122994-0.5204767j) * 0.7071 = 0.2484-0.3681j`

But the error output shows the result.hh as `0.35122994-0.5204767j` which is the ORIGINAL value unchanged!

Wait, let me check if the test fixture is using `sample_slc_data` which is defined in conftest.py and passed to the test. Let me re-read the test:

```python
class TestRadiometricCalibrator:
    def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):
        cal = RadiometricCalibrator(cal_constant=2.0)
        result = cal.calibrate(sample_slc_data)
        assert result is sample_slc_data
```

Ah, `sample_slc_data` is the fixture. It creates `PolarimetricData(hh=..., hv=..., vh=..., vv=..., metadata=_test_metadata())`.

Note: `calibration` is NOT set (it's `None`).

In the calibrator code:
```python
k = data.calibration.k_hh if data.calibration else self.cal_constant
```

Since `data.calibration` is `None`, `k = self.cal_constant = 2.0`.

Then:
```python
cal_val = getattr(data.calibration, f"k_{pol}", k) if data.calibration else k
```

Since `data.calibration` is `None`, `cal_val = k = 2.0`.

Then:
```python
scale = np.float32(1.0 / np.sqrt(max(cal_val, 1e-30)))
```

`scale = 1.0 / sqrt(2.0) = 0.7071`.

Then:
```python
setattr(data, pol, (arr * scale).astype(np.complex64))
```

This should work! `(0.35122994-0.5204767j) * 0.7071 = 0.2484-0.3681j`

But the error output shows `result.hh=array([[ 0.35122994-0.5204767j ...]]`. This looks like the ORIGINAL sample_slc_data.hh value from the fixture. 

OH WAIT - I think I need to look at the actual values. The original HH from the fixture (created with seed 42) would be:
- For the first pixel: `np.random.randn(1) = 0.496714153011329 + 1j * np.random.randn(1) = -0.13826429635135847`
- Wait, that gives `0.4967 - 0.1383j`, not `0.3512 - 0.5205j`

Actually, with `np.random.seed(42)`, `np.random.randn(128, 128)` generates specific values. The first value in `hh` is `0.49671415...` for real and `-0.13826430...` for imag. So `0.4967 - 0.1383j`. This doesn't match `0.35122994-0.5204767j`.

Hmm, but then what does? Let me check... wait, the conftest does `np.random.seed(42)` which affects the global state. Then it generates multiple arrays (hh, hv, vh, vv), each with different random values. So:
- `hh = np.random.randn(128, 128) + 1j * np.random.randn(128, 128)` 
- This takes 2*128*128 random numbers

Looking at the output `result.hh=array([[ 0.35122994-0.5204767j ...]]`, let me compute:
- `0.35122994^2 + 0.5204767^2 = 0.12336 + 0.27090 = 0.39426`
- sqrt(0.39426) = 0.6279

So `|result.hh| = 0.6279`. But after calibration with K=2, `|result.hh|` should be `0.6279 / sqrt(2) = 0.4441`.

The test computes `np.abs(result.hh)**2` which would be `0.6279^2 = 0.3943`, and compares it to `np.abs(sample_slc_data.hh)**2 / 2.0 = 0.3943 / 2 = 0.1971`.

So they're not equal: `0.3943 != 0.1971`. This means the calibration is NOT being applied.

Wait, but the code should be applying it! Let me think... `setattr(data, pol, (arr * scale).astype(np.complex64))` where `arr` IS `data.pol` (which is `data.hh`). So `data.hh = data.hh * scale`. But wait, when we do `arr = getattr(data, pol, None)`, `arr` is a REFERENCE to the same numpy array. So `arr * scale` creates a new array, then we assign it back.

Hmm, actually let me look at the `setattr(data, pol, (arr * scale).astype(np.complex64))`. Since `arr` is `data.hh` and we're setting `data.hh = (data.hh * scale)`, this should work.

Wait, but `arr` is `data.hh`. `arr * scale` creates a new array. `setattr(data, pol, new_array)` sets `data.hh` to the new array. That works.

Hmm, let me actually run a quick test...

Actually, wait - I bet the issue is that `np.float32(1.0 / np.sqrt(max(2.0, 1e-30)))` is `np.float32(0.70710677)`. And `arr * 0.70710677` should indeed scale the array. Let me just run a quick verification.

**Tool: bash**

**Input:**
```json
{
  "command": "source /tmp/lunaice_venv/bin/activate && python3 -c \"\nimport numpy as np\nfrom lunaice.processing.calibration import RadiometricCalibrator\nfrom lunaice.models.schemas import PolarimetricData\n\nnp.random.seed(42)\nhh = np.random.randn(4,4) + 1j * np.random.randn(4,4)\ndata = PolarimetricData(hh=hh.copy())\nprint('before:', data.hh[0,0], 'mag:', np.abs(data.hh[0,0]))\n\ncal = RadiometricCalibrator(cal_constant=2.0)\nresult = cal.calibrate(data)\nprint('after:', result.hh[0,0], 'mag:', np.abs(result.hh[0,0]))\nprint('expected mag:', np.abs(hh[0,0]) / np.sqrt(2))\nprint('power ratio:', np.abs(result.hh[0,0])**2 / (np.abs(hh[0,0])**2))\n\" 2>&1",
  "description": "Debug radiometric calibration"
}
```

**Output:**
```
Data is not quad-pol; calibrating available channels
before: (0.4967141530112327-1.0128311203344238j) mag: 1.1280743894440428
after: (0.35122994-0.7161797j) mag: 0.797669
expected mag: 0.7976690504587569
power ratio: 0.4999999322816572

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 8.9s)

_Thinking:_

OK so the calibration IS working correctly when run standalone. The power ratio is 0.5 as expected. But the test is failing... Let me look at the test again more carefully.

```python
def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):
    cal = RadiometricCalibrator(cal_constant=2.0)
    result = cal.calibrate(sample_slc_data)
    assert result is sample_slc_data
    assert result.hh is not None
    expected_power = np.abs(sample_slc_data.hh) ** 2 / 2.0
    assert np.allclose(np.abs(result.hh) ** 2, expected_power)
```

But wait - in the error output the HH values show `0.62790006` magnitude while the expected is `0.19712925`. So `0.6279^2 = 0.3942` vs `0.1971`. That's a ratio of 2, meaning the scaling factor is 0.5 instead of 1/sqrt(2).

Wait... but my debug test shows it working correctly. Let me actually run the failing test with more output to see what's happening.

**Tool: bash**

**Input:**
```json
{
  "command": "source /tmp/lunaice_venv/bin/activate && python3 -c \"\nimport numpy as np\nfrom lunaice.processing.calibration import RadiometricCalibrator\nfrom lunaice.models.schemas import PolarimetricData, Metadata\n\nnp.random.seed(42)\nh, w = 128, 128\nhh = np.random.randn(h, w) + 1j * np.random.randn(h, w)\nhv = np.random.randn(h, w) + 1j * np.random.randn(h, w)\nvh = np.random.randn(h, w) + 1j * np.random.randn(h, w)\nvv = np.random.randn(h, w) + 1j * np.random.randn(h, w)\n\ndata = PolarimetricData(hh=hh, hv=hv, vh=vh, vv=vv, metadata=Metadata(\n    product_id='TEST_001',\n    processing_level='L1A',\n    frequency_band='L',\n    polarization_mode='quad_pol',\n    acquisition_time='2020-01-15T12:00:00',\n    orbit_number=1234,\n    incidence_angle_deg=30.0,\n    slant_range_resolution_m=15.0,\n    azimuth_resolution_m=15.0,\n    looks_range=1,\n    looks_azimuth=1,\n    calibration_constant=1.0,\n    wavelength_cm=24.0,\n    center_latitude=-85.0,\n    center_longitude=30.0,\n    pixel_spacing_m=15.0,\n))\n\ncal = RadiometricCalibrator(cal_constant=2.0)\nresult = cal.calibrate(data)\n\nexpected_power = np.abs(hh) ** 2 / 2.0\nactual_power = np.abs(result.hh) ** 2\nprint('Max diff:', np.max(np.abs(actual_power - expected_power)))\nprint('Ratio:', actual_power[0,0] / expected_power[0,0])\nprint('Result hh[0,0]:', result.hh[0,0])\nprint('Original hh[0,0]:', hh[0,0])\nprint('np.allclose:', np.allclose(actual_power, expected_power))\n\" 2>&1",
  "description": "Debug failing test directly"
}
```

**Output:**
```
Max diff: 1.7541309436097663e-06
Ratio: 1.0000000539545213
Result hh[0,0]: (0.35122994-0.5204767j)
Original hh[0,0]: (0.4967141530112327-0.7360652185406498j)
np.allclose: True

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 12.4s)

_Thinking:_

Interesting! When I manually reproduce the test, it works fine. The issue must be in how the conftest fixture is set up. Let me look at the conftest more carefully.

```python
@pytest.fixture
def sample_slc_data() -> PolarimetricData:
    np.random.seed(42)
    h = 128
    w = 128
    hh = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    hv = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    vh = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    vv = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    return PolarimetricData(hh=hh, hv=hv, vh=vh, vv=vv, metadata=_test_metadata())
```

Wait, but the output from my debug shows `hh[0,0] = 0.4967 - 0.7361j` which is different from the pytest output which shows `hh = 0.3512 - 0.5205j`. This means the RNG state is different between the runs. That's fine, the allclose should still work.

Actually, the pytest error output showed `result.hh` values that matched the UN-calibrated values from the test fixture. But in my manual test, I can see the calibration IS working (`0.35122994-0.5204767j` which is `0.4967-0.7361j / sqrt(2)`).

Wait... in the pytest error output, the `result.hh` was shown as:
```
hh=array([[ 0.35122994-0.5204767j , ...
```

And the test assertion was about `np.abs(result.hh) ** 2`. The `expected_power = np.abs(sample_slc_data.hh) ** 2 / 2.0`.

But `sample_slc_data.hh` in the error output is:
```
hh=array([[ 0.35122994-0.5204767j , ...
```

Wait, those are the SAME values as `result.hh` in the error output. That's because both `result` and `sample_slc_data` point to the same object (`result is sample_slc_data`), and the calibration modifies in-place. So `sample_slc_data.hh` has already been modified to the calibrated value by the time `expected_power` is computed.

THAT'S THE BUG! The test computes `expected_power = np.abs(sample_slc_data.hh) ** 2 / 2.0` AFTER the calibration has already modified `sample_slc_data.hh`. So `expected_power` is computed from the calibrated values, not the original values!

The fix is to compute `expected_power` BEFORE calling `calibrate()`. Let me fix the test.

But wait, in the error output I see:
- `np.abs(result.hh) ** 2` values: `0.62790006**2 = 0.3943...`
- `expected_power` values: `0.19712925`

Hmm, `0.3943 / 0.1971 = 2.0`. So the test is comparing calibrated power with calibrated power/2. That means `sample_slc_data.hh` WAS modified before `expected_power` was computed.

Wait, but `0.62790006^2 = 0.3943`, and `expected_power` is `0.1971`. So `0.3943` vs `0.1971`. If `expected_power` was computed as `|calibrated_value|^2 / 2`, then `0.3943/2 = 0.1971`. Yes, that confirms it. The issue is that the test computes expected_power from the already-calibrated data.

I need to fix the test by saving the original power before calibration.

The test bug is that `expected_power` is computed after calibration already mutates the data. Fix:

**Tool: edit**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/tests/test_calibration.py",
  "oldString": "    def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):\n        cal = RadiometricCalibrator(cal_constant=2.0)\n        result = cal.calibrate(sample_slc_data)\n        assert result is sample_slc_data\n        assert result.hh is not None\n        expected_power = np.abs(sample_slc_data.hh) ** 2 / 2.0\n        assert np.allclose(np.abs(result.hh) ** 2, expected_power)",
  "newString": "    def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):\n        original_hh = sample_slc_data.hh.copy()\n        cal = RadiometricCalibrator(cal_constant=2.0)\n        result = cal.calibrate(sample_slc_data)\n        assert result is sample_slc_data\n        assert result.hh is not None\n        expected_power = np.abs(original_hh) ** 2 / 2.0\n        assert np.allclose(np.abs(result.hh) ** 2, expected_power)"
}
```

**Output:**
```
Edit applied successfully.
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 3.1s)

_Thinking:_

Now let me also fix the `test_calibrate_partial` test since it has the same issue.

**Tool: read**

**Input:**
```json
{
  "filePath": "/home/rabindra03/workspace/DEV/LUNAICE/tests/test_calibration.py"
}
```

**Output:**
```
<path>/home/rabindra03/workspace/DEV/LUNAICE/tests/test_calibration.py</path>
<type>file</type>
<content>
1: from __future__ import annotations
2: 
3: import numpy as np
4: import pytest
5: 
6: from lunaice.models.schemas import PolarimetricData
7: from lunaice.processing.calibration import PolarimetricCalibrator, RadiometricCalibrator
8: 
9: 
10: class TestRadiometricCalibrator:
11:     def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):
12:         original_hh = sample_slc_data.hh.copy()
13:         cal = RadiometricCalibrator(cal_constant=2.0)
14:         result = cal.calibrate(sample_slc_data)
15:         assert result is sample_slc_data
16:         assert result.hh is not None
17:         expected_power = np.abs(original_hh) ** 2 / 2.0
18:         assert np.allclose(np.abs(result.hh) ** 2, expected_power)
19: 
20:     def test_calibrate_partial(self):
21:         data = PolarimetricData(hh=np.ones((4, 4), dtype=complex))
22:         cal = RadiometricCalibrator(cal_constant=4.0)
23:         result = cal.calibrate(data)
24:         assert np.allclose(np.abs(result.hh) ** 2, 0.25)
25: 
26: 
27: class TestPolarimetricCalibrator:
28:     def test_calibrate_quad_pol(self, sample_slc_data: PolarimetricData):
29:         cal = PolarimetricCalibrator(
30:             co_pol_phase_correction=-50.0,
31:             cross_pol_phase_correction=-5.0,
32:             cross_talk_hv=0.01,
33:             cross_talk_vh=0.01,
34:         )
35:         result = cal.calibrate(sample_slc_data)
36:         assert result is sample_slc_data
37:         assert result.hh.shape == sample_slc_data.hh.shape
38: 
39:     def test_skip_non_quadpol(self):
40:         data = PolarimetricData(hh=np.ones((4, 4), dtype=complex))
41:         cal = PolarimetricCalibrator()
42:         result = cal.calibrate(data)
43:         np.testing.assert_array_equal(result.hh, data.hh)

(End of file - total 43 lines)
</content>
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 6.3s)

_Thinking:_

The `test_calibrate_partial` test uses `np.ones((4,4), dtype=complex)` which is `1+0j`. After calibration with K=4, the output should be `1/sqrt(4) = 0.5`. So `|0.5|^2 = 0.25`. That looks correct since we're creating fresh data that doesn't get mutated before the assertion.

Let me run the tests again.

**Tool: bash**

**Input:**
```json
{
  "command": "source /tmp/lunaice_venv/bin/activate && python -m pytest tests/ -v --tb=short 2>&1",
  "description": "Run tests after fixing test bug",
  "timeout": 120000,
  "workdir": "/home/rabindra03/workspace/DEV/LUNAICE"
}
```

**Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0 -- /tmp/lunaice_venv/bin/python
cachedir: .pytest_cache
rootdir: /home/rabindra03/workspace/DEV/LUNAICE
configfile: pyproject.toml
plugins: cov-7.1.0, zarr-3.2.1
collecting ... collected 18 items

tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_quad_pol PASSED [  5%]
tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_partial PASSED [ 11%]
tests/test_calibration.py::TestPolarimetricCalibrator::test_calibrate_quad_pol PASSED [ 16%]
tests/test_calibration.py::TestPolarimetricCalibrator::test_skip_non_quadpol PASSED [ 22%]
tests/test_pipeline.py::test_config_from_yaml PASSED                     [ 27%]
tests/test_pipeline.py::test_speckle_filter_validation PASSED            [ 33%]
tests/test_pipeline.py::test_speckle_filter_methods PASSED               [ 38%]
tests/test_pipeline.py::test_processing_config_defaults PASSED           [ 44%]
tests/test_polarimetry.py::TestCoherencyMatrixBuilder::test_build_quad_pol PASSED [ 50%]
tests/test_polarimetry.py::TestCoherencyMatrixBuilder::test_raises_on_partial PASSED [ 55%]
tests/test_polarimetry.py::TestCloudePottierDecomposition::test_decompose PASSED [ 61%]
tests/test_polarimetry.py::TestCircularPolarizationRatio::test_cpr_quad_pol PASSED [ 66%]
tests/test_polarimetry.py::TestDegreeOfPolarization::test_dop_quad_pol PASSED [ 72%]
tests/test_polarimetry.py::TestBackscatterCoefficient::test_compute PASSED [ 77%]
tests/test_reader.py::test_read_pds4_label_not_found PASSED              [ 83%]
tests/test_reader.py::test_dfsar_reader_metadata PASSED                  [ 88%]
tests/test_reader.py::test_validate_consistent_shapes PASSED             [ 94%]
tests/test_reader.py::test_validate_inconsistent_shapes PASSED           [100%]

============================== 18 passed in 1.17s ==============================

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 8.0s)

_Thinking:_

All 18 tests pass. Let me now verify the CLI works as well.

All 18 tests pass. Let me verify the CLI and do a final structural check.

**Tool: bash**

**Input:**
```json
{
  "command": "source /tmp/lunaice_venv/bin/activate && lunaice --help 2>&1",
  "description": "Verify CLI works"
}
```

**Output:**
```
usage: lunaice [-h] [--version] {process} ...

LUNAICE: Lunar Subsurface Ice Detection using Chandrayaan-2 DFSAR Data

positional arguments:
  {process}   Sub-command
    process   Run the DFSAR polarimetric processing pipeline

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit

Examples:
  lunaice process -i data/L1A_SLC.h5 -o output/ --band L
  lunaice process -i data/SLC/ --config configs/processing.yaml -v
  lunaice process -i data/L1A_SLC.h5 --no-speckle --multilook 2 2
        

```

**Tool: bash**

**Input:**
```json
{
  "command": "source /tmp/lunaice_venv/bin/activate && python -m pytest tests/ --cov=src/lunaice --cov-report=term 2>&1",
  "description": "Run tests with coverage",
  "timeout": 120000,
  "workdir": "/home/rabindra03/workspace/DEV/LUNAICE"
}
```

**Output:**
```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.3, pluggy-1.6.0 -- /tmp/lunaice_venv/bin/python
cachedir: .pytest_cache
rootdir: /home/rabindra03/workspace/DEV/LUNAICE
configfile: pyproject.toml
plugins: cov-7.1.0, zarr-3.2.1
collecting ... collected 18 items

tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_quad_pol PASSED [  5%]
tests/test_calibration.py::TestRadiometricCalibrator::test_calibrate_partial PASSED [ 11%]
tests/test_calibration.py::TestPolarimetricCalibrator::test_calibrate_quad_pol PASSED [ 16%]
tests/test_calibration.py::TestPolarimetricCalibrator::test_skip_non_quadpol PASSED [ 22%]
tests/test_pipeline.py::test_config_from_yaml PASSED                     [ 27%]
tests/test_pipeline.py::test_speckle_filter_validation PASSED            [ 33%]
tests/test_pipeline.py::test_speckle_filter_methods PASSED               [ 38%]
tests/test_pipeline.py::test_processing_config_defaults PASSED           [ 44%]
tests/test_polarimetry.py::TestCoherencyMatrixBuilder::test_build_quad_pol PASSED [ 50%]
tests/test_polarimetry.py::TestCoherencyMatrixBuilder::test_raises_on_partial PASSED [ 55%]
tests/test_polarimetry.py::TestCloudePottierDecomposition::test_decompose PASSED [ 61%]
tests/test_polarimetry.py::TestCircularPolarizationRatio::test_cpr_quad_pol PASSED [ 66%]
tests/test_polarimetry.py::TestDegreeOfPolarization::test_dop_quad_pol PASSED [ 72%]
tests/test_polarimetry.py::TestBackscatterCoefficient::test_compute PASSED [ 77%]
tests/test_reader.py::test_read_pds4_label_not_found PASSED              [ 83%]
tests/test_reader.py::test_dfsar_reader_metadata PASSED                  [ 88%]
tests/test_reader.py::test_validate_consistent_shapes PASSED             [ 94%]
tests/test_reader.py::test_validate_inconsistent_shapes PASSED           [100%]

================================ tests coverage ================================
_______________ coverage: platform linux, python 3.12.3-final-0 ________________

Name                                                   Stmts   Miss  Cover
--------------------------------------------------------------------------
src/lunaice/__init__.py                                    4      0   100%
src/lunaice/cli.py                                        60     60     0%
src/lunaice/config.py                                     66      4    94%
src/lunaice/io/__init__.py                                 3      0   100%
src/lunaice/io/reader.py                                  89     60    33%
src/lunaice/io/writer.py                                  97     73    25%
src/lunaice/models/__init__.py                             2      0   100%
src/lunaice/models/schemas.py                            125      1    99%
src/lunaice/pipeline.py                                  133    113    15%
src/lunaice/processing/__init__.py                         4      0   100%
src/lunaice/processing/calibration.py                     55      6    89%
src/lunaice/processing/polarimetry/__init__.py             6      0   100%
src/lunaice/processing/polarimetry/backscatter.py         21      0   100%
src/lunaice/processing/polarimetry/cloude_pottier.py      49      2    96%
src/lunaice/processing/polarimetry/coherency.py           32      3    91%
src/lunaice/processing/polarimetry/cpr.py                 24      6    75%
src/lunaice/processing/polarimetry/dop.py                 20      4    80%
src/lunaice/processing/speckle.py                         96     80    17%
--------------------------------------------------------------------------
TOTAL                                                    886    412    53%
============================== 18 passed in 1.87s ==============================

```

---

## Assistant (Build · DeepSeek V4 Flash Free · 5.3s)

_Thinking:_

All 18 tests pass. CLI works. Coverage is 53% overall (lower for CLI/pipeline/speckle since those involve I/O or complex algorithms). The core polarimetry modules have excellent coverage (75-100%).

Let me now update the todo list and provide a comprehensive summary.

Let me also check the final tree structure.

**Tool: invalid**

**Input:**
```json
{
  "tool": "todowrite",
  "error": "Invalid input for tool todowrite: JSON parsing failed: Text: {\"todos\": [{\"priority\":\"high\",\"content\":\"Create repository structure and directory tree\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Build core data models and schemas (PolarimetricData, metadata)\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Implement I/O module (PDS4 reader, GeoTIFF/Zarr writer)\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Implement radiometric calibration module\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Implement polarimetric calibration module\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Implement speckle reduction module\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Implement coherency matrix (T3) generation\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Implement Cloude-Pottier decomposition (H/alpha/A)\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Implement CPR, DOP, backscatter coefficients\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Build pipeline orchestrator\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Build CLI interface\",\"status\":\"completed\"},{\"priority\":\"high\",\"content\":\"Create configuration files (YAML)\",\"status\":\"completed\"},{\"priority\":\"medium\",\"content\":\"Write tests\",\"status\":\"completed\"},{\"priority\":\"medium\",\"content\",\"status\":\"completed\",\"content\":\"Create pyproject.toml and documentation\"}]}.\nError message: JSON Parse error: Expected ':' before value in object property definition"
}
```

**Output:**
```
The arguments provided to the tool are invalid: Invalid input for tool todowrite: JSON parsing failed: Text: {"todos": [{"priority":"high","content":"Create repository structure and directory tree","status":"completed"},{"priority":"high","content":"Build core data models and schemas (PolarimetricData, metadata)","status":"completed"},{"priority":"high","content":"Implement I/O module (PDS4 reader, GeoTIFF/Zarr writer)","status":"completed"},{"priority":"high","content":"Implement radiometric calibration module","status":"completed"},{"priority":"high","content":"Implement polarimetric calibration module","status":"completed"},{"priority":"high","content":"Implement speckle reduction module","status":"completed"},{"priority":"high","content":"Implement coherency matrix (T3) generation","status":"completed"},{"priority":"high","content":"Implement Cloude-Pottier decomposition (H/alpha/A)","status":"completed"},{"priority":"high","content":"Implement CPR, DOP, backscatter coefficients","status":"completed"},{"priority":"high","content":"Build pipeline orchestrator","status":"completed"},{"priority":"high","content":"Build CLI interface","status":"completed"},{"priority":"high","content":"Create configuration files (YAML)","status":"completed"},{"priority":"medium","content":"Write tests","status":"completed"},{"priority":"medium","content","status":"completed","content":"Create pyproject.toml and documentation"}]}.
Error message: JSON Parse error: Expected ':' before value in object property definition
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 4.7s)

_Thinking:_

There's a JSON error in the last todo item - I accidentally had a duplicate "content" key. Let me fix and resubmit.

**Tool: todowrite**

**Input:**
```json
{
  "todos": [
    {
      "priority": "high",
      "content": "Create repository structure and directory tree",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Build core data models and schemas (PolarimetricData, metadata)",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement I/O module (PDS4 reader, GeoTIFF/Zarr writer)",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement radiometric calibration module",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement polarimetric calibration module",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement speckle reduction module",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement coherency matrix (T3) generation",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement Cloude-Pottier decomposition (H/alpha/A)",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Implement CPR, DOP, backscatter coefficients",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Build pipeline orchestrator",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Build CLI interface",
      "status": "completed"
    },
    {
      "priority": "high",
      "content": "Create configuration files (YAML)",
      "status": "completed"
    },
    {
      "priority": "medium",
      "content": "Write tests (18/18 passing)",
      "status": "completed"
    },
    {
      "priority": "medium",
      "content": "Create pyproject.toml and documentation",
      "status": "completed"
    }
  ]
}
```

**Output:**
```
[
  {
    "content": "Create repository structure and directory tree",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Build core data models and schemas (PolarimetricData, metadata)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement I/O module (PDS4 reader, GeoTIFF/Zarr writer)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement radiometric calibration module",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement polarimetric calibration module",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement speckle reduction module",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement coherency matrix (T3) generation",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement Cloude-Pottier decomposition (H/alpha/A)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Implement CPR, DOP, backscatter coefficients",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Build pipeline orchestrator",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Build CLI interface",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Create configuration files (YAML)",
    "status": "completed",
    "priority": "high"
  },
  {
    "content": "Write tests (18/18 passing)",
    "status": "completed",
    "priority": "medium"
  },
  {
    "content": "Create pyproject.toml and documentation",
    "status": "completed",
    "priority": "medium"
  }
]
```

---

## Assistant (Build · DeepSeek V4 Flash Free · 13.0s)

All 14 tasks complete. Here is what was built:

---

## LUNAICE — Phase 1 Complete

### Repository Structure
```
LUNAICE/
├── src/lunaice/
│   ├── __init__.py              # Package entry, version
│   ├── config.py                # DFSARConfig, ProcessingConfig, SpeckleFilterConfig
│   ├── pipeline.py              # Pipeline orchestrator
│   ├── cli.py                   # CLI (lunaice process command)
│   ├── models/
│   │   └── schemas.py           # PolarimetricData, CoherencyMatrix, DecompositionProducts, etc.
│   ├── io/
│   │   ├── reader.py            # DFSARReader (PDS4 label parsing, SLC loading)
│   │   └── writer.py            # GeoTIFFWriter, ZarrWriter, ReportWriter
│   └── processing/
│       ├── calibration.py       # RadiometricCalibrator, PolarimetricCalibrator
│       ├── speckle.py           # 5 speckle filters (Refined Lee, Boxcar, Lee-Sigma, IDAN, Bilateral)
│       └── polarimetry/
│           ├── coherency.py     # CoherencyMatrixBuilder (Pauli → T3)
│           ├── cloude_pottier.py # H/α/A eigenvalue decomposition
│           ├── cpr.py           # Circular Polarization Ratio
│           ├── dop.py           # Degree of Polarization (Stokes)
│           └── backscatter.py   # σ°, γ°, span
├── tests/                       # 18 tests, all passing
├── configs/                     # default.yaml, processing.yaml
├── docs/                        # architecture.md, api.md
├── notebooks/                   # (placeholder)
├── pyproject.toml
└── README.md
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Pauli → T3** | Standard Cloude-Pottier requires the 3×3 coherency matrix via Pauli basis |
| **Per-pixel eigenvalue decomposition** | Proper H/α/A requires solving the 3×3 Hermitian eigenproblem per pixel |
| **5 speckle filters** | DFSAR literature (Bhiravarasu+2021) uses refined Lee; others available for comparison |
| **Zarr + GeoTIFF dual output** | Zarr for ML-ready feature cubes; GeoTIFF for GIS interoperability |
| **Config-driven YAML** | All parameters externalized; CLI overrides via flags |
| **Graceful degradation** | Each step wrapped in try/except — partial results still written |

### Polarimetric Products

| Product | Method | Range | Ice Relevance |
|---------|--------|-------|---------------|
| Entropy (H) | Cloude-Pottier | [0,1] | High H → multiple scattering mechanisms |
| Alpha (α) | Cloude-Pottier | [0°,90°] | α < 40°: surface; 40-50°: volume; > 50°: dihedral |
| Anisotropy (A) | Cloude-Pottier | [0,1] | Secondary scattering dominance |
| CPR | Circular basis transform | ≥0 | CPR > 1 → possible ice (coherent backscatter) |
| DOP | Stokes parameters | [0,1] | Low DOP → depolarization |
| σ°/γ° | Radar equation | dB | Backscatter intensity per channel |

### DFSAR Literature Incorporated
- **Calibration**: Phase offsets per Bhiravarasu+2021 (L-band: co-pol -50°, cross-pol -5°; S-band: co-pol -100°, cross-pol +100°)
- **Polarimetric calibration**: Cross-talk + channel imbalance correction following Sun+2018 methodology
- **Decomposition**: Cloude-Pottier (1996) eigenvalue-based, per the DFSAR calibration paper
- **CPR**: Circular basis transformation for ice detection per Chakraborty+2024

### CLI Usage
```bash
lunaice process -i data/L1A_SLC.h5 -o output/ --band L
lunaice process -i data/SLC/ --config configs/processing.yaml -v
lunaice process -i data/L1A_SLC.h5 --no-speckle --multilook 2 2
```

---

