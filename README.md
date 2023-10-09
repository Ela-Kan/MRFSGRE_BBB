# MRFSGRE_BBB: MR Fingerprinting for Blood Brain Barrier Disruption

-----------------------------------------------------------------------------
Author: Emma Thomson \
Year: 2020-2023 \
Institution: Centre for Medical Image Computing: University College London \
Email: thomson.l.emma@gmail.com 

-----------------------------------------------------------------------------
Bloch simulation code for a two compartment model with a semipermeable barrier 

INLUDING:
- 2D set of isochromats with frequency encoding 
- Exchange between two compartments
- Variable fractional compartment sizes  
- Variable permeability of barrier 
- Slice profile correction

IMPORTANT ASSUMPTIONS
 - The effects of relaxation are ignored during the RF pulses 
 - Between pulses all transverse magnetization has relaxed back to
   equilibrium
 - Frequencies in MHz have been converted to Hz to make the simulation
   easier to visualise

------------------------------------------------------------------------------
## USER NOTES: 

### To generate dictionaries: 

- Run dictionaryGeneration.py (or dictionaryGenerationGPU.py if you want to run on GPUs)
- Code within the functions folder contains the functions called by the dictionary generation code and does not need to be accessed (unless you wish to make fundamental changes to the simulation) 
- The dictionary generation code saves the signals in a folder within ./Dictionaries. Run the conversion code (dictionaryMatching/dictionaryConversion.py) in order to compress the dictionary into one dictionary.txt file and one lookupTable.txt file 
- The slice profile array used to calculate slice profile effects is provided in sliceProfile/SliceProfile.mat. If you wish to regenerate the slice profile, you will need to use the Multiband RF toolbox repository [1]
- An example 5D dictionary can be found on my dropbox [*] 

###To generate noisy signals:

- Use the dictionary generation code as above, but change the 'noise' parameter (indicates the number of noise levels calculated) and 'samples' parameter (number of times each noise level is recalculated) 
- These signals are matched through the the dictionaryMatching/simulationMatching.py code and the arrays of matched parameters stored in Dictionaries/SimulationMatching/Matching[DictionaryName]
- To plot these results as quiver plots associated with these simulations run the plottingAndStatistics/simulationGridPlotting.py code
- A 2D and 5D dictionary of noisy signals at 10 noise levels can be found on my dropbox [*]


### To generate quantitative maps: 

- To generate quantitative maps with the standard matching as described in Ma et al. [2] run the dictionaryMatching/standardMatching.py code. This code generates and saves the five quantitative maps to VolunteerX/MRF/Maps/. 
- To generate maps using the novel 'B1-first' matching technique run dictionaryMatching/b1FirstMatching.py. Bar the B1 matching changes this code performs the same way as the standard matching code
- A single volunteer data set is provided (SampleData) for your testing purposes 
- In order to view these maps in a single figure use the plotting code (plottingAndStatistics/quantativeMaps.py)

### To perform statistics:

- Run all the statistics from the plottingAndStatistics/statistics.py code. This code calculates the repeatability coefficient and intraclass correlation coefficient between scan and rescan data, Bland-Altmann plots, bar chart for mean parameter values for white and grey matter, and performs t-tests for these means. 

------------------------------------------------------------------------------
[*] For example dictionaries please visit [my Dropbox](https://www.dropbox.com/scl/fo/l7prpa1sz44fc5wvdpsut/h?rlkey=hnj0z6w59lwwm25w1mvrjmcnq&dl). 

If comparing to the relevant publication: 
- 2D simulation dictionary = Dictionary2Dsim
- 5D simulation dictionary = Dictionary5Dsim (warning: this dictionary is over 200GB)
- Dictionary for experimental data = DictionaryLarge

------------------------------------------------------------------------------

[1] https://github.com/mriphysics/Multiband-RF \
[2] Ma Dan, Gulani Vikas, Seiberlich Nicole, et al. Magnetic Resonance Fingerprinting. Nature. 2013;495:187-192
