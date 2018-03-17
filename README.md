<h1 style='text-align:center'>pubsne</h1>
<h3 style='text-align:center'>Public Supernova Probject</h3>

<h2 style='text-align:center'>General Layout of the Module</h2>

<h4>scraper.py</h4>

<h4>creator.py</h4>

<h4>base.py</h4>

<h2 style='text-align:center'>Dependencies: Other parts of the Module</h2>

<h4>snmc.py</h4>

<h4>snetools.py</h4>

<h4>tools.py</h4>

<h4>mfa.py</h4>

<h4>IDRTools.py</h4>

<h2 style='text-align:center'>Approximate Run-Times</h2>

Note that all run-times were from my laptop last time I attempted to execute them. My laptop's processor is:

Intel® Core™ i5-4210U CPU @ 1.70GHz × 4 

<h4>scraper.py</h4>

The scraper.py file has a main method implemented that does the full extraction (assuming an 'sne.csv' file exists in the same directory) from sne.space. So if you just copy this file and then run it, it will execute immediately.

Run-time: ~30 minutes.

<h4>creator.py</h4>

The creator.py has a main method implemented that does the object creation already. So if you just copy this file and then run it, it will execute immediately. It is set to create all the pickled files containing useful dictionaries with various objects in them (explained earlier).

NOTE: if you do not have Sam's IDRTools.py and the snfidrtools module, then you will get errors executing this script unless you specify within the code to only work with public sources.
To only use public sources, change the line 201 in the code to:
	"for source in ['Public']:"

Run-time: ~1.3 minutes.

<h4>base.py</h4>

The base.py module does not have a main method implemented because there are several things you could do. I will outline the run-times of the three biggest things here.

NOTE: currently, each of the methods is set to only run through combinations of 'Factory' and 'Public' training/fitting. To run through all possible combinations (that is, adding in 'All' and 'Bsnip'), replace '.main_sources' with '.sources' everywhere.

create_web_gallery(): ~11.9 minutes.
make_white_noise_parallel(): ~1.7 minutes.
Abase.group_explained_variance_plots(): ~1.4 minutes
Abase.pub_comp_breakdown(): ~0.9 minutes
