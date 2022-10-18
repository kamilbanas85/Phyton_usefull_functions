https://towardsdatascience.com/bootstrapping-vs-permutation-testing-a30237795970


permuttion_replcates

https://thomasleeper.com/Rcourse/Tutorials/permutationtests.html   :
Permutation Tests
An increasingly common statistical tool for constructing sampling distributions is the permutation test (or sometimes called a randomization test). Like bootstrapping, a permutation test builds - rather than assumes - sampling distribution (called the “permutation distribution”) by resampling the observed data. Specifically, we can “shuffle” or permute the observed data (e.g., by assigning different outcome values to each observation from among the set of actually observed outcomes). Unlike bootstrapping, we do this without replacement.

Permutation tests are particularly relevant in experimental studies, where we are often interested in the sharp null hypothesis of no difference between treatment groups. In these situations, the permutation test perfectly represents our process of inference because our null hypothesis is that the two treatment groups do not differ on the outcome (i.e., that the outcome is observed independently of treatment assignment). When we permute the outcome values during the test, we therefore see all of the possible alternative treatment assignments we could have had and where the mean-difference in our observed data falls relative to all of the differences we could have seen if the outcome was independent of treatment assignment. While a permutation test requires that we see all possible permutations of the data (which can become quite large), we can easily conduct “approximate permutation tests” by simply conducting a vary large number of resamples. That process should, in expectation, approximate the permutation distribution.

For example, if we have only n=20 units in our study, the number of permutations is:



https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Permutation%20Tests_final.pdf


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
https://www.jwilber.me/permutationtest/



Before hypotheis (permutetaion test) draw a ECDF ot the samples:
! I encourage you to go ahead and plot the ECDFs right now.
 You will see by eye that the null hypothesis that the distributions are the same is almost certainly not true.



#############################################################################
#############################################################################
### correlation

## to check if correlation is true it is posible to make hypothesis test (permutaion test) i which persson correlation is a test statistic
## ( f.e. cor = 0.54 if is it true or by a chance). Null hypohesis: 2 variables are uncorelated



###############################################################
### ecdf
# Compute ECDFs
x_1975, y_1975 = ecdf(bd_1975)
x_2012, y_2012 = ecdf(bd_2012)

# Plot the ECDFs
_ = plt.plot(x_1975, y_1975, marker='.', linestyle='none')
_ = plt.plot(x_2012, y_2012, marker='.', linestyle='none')

# Set margins
plt.margins(0.02)

# Add axis labels and legend
_ = plt.xlabel('beak depth (mm)')
_ = plt.ylabel('ECDF')
_ = plt.legend(('1975', '2012'), loc='lower right')

# Show the plot
plt.show()


##################################################
### 95% confidence interval based on bootstraping

# Compute the difference of the sample means: mean_diff
mean_diff = np.mean(bd_2012) - np.mean(bd_1975)

# Get bootstrap replicates of means
bs_replicates_1975 = draw_bs_reps(bd_1975, np.mean, 10000)
bs_replicates_2012 = draw_bs_reps(bd_2012, np.mean, 10000)

# Compute samples of difference of means: bs_diff_replicates
bs_diff_replicates = bs_replicates_2012 - bs_replicates_1975

# Compute 95% confidence interval: conf_int
conf_int = np.percentile(bs_diff_replicates, [2.5, 97.5])


#### or 

# Compute the Pearson correlation coefficients
r_scandens = pearson_r(bd_parent_scandens, bd_offspring_scandens)
r_fortis = pearson_r(bd_parent_fortis,  bd_offspring_fortis)

# Acquire 1000 bootstrap replicates of Pearson r
bs_replicates_scandens = draw_bs_pairs(bd_parent_scandens, bd_offspring_scandens, pearson_r, 1000)

bs_replicates_fortis = draw_bs_pairs(bd_parent_fortis,  bd_offspring_fortis, pearson_r, 1000)


# Compute 95% confidence intervals
conf_int_scandens = np.percentile(bs_replicates_scandens, [2.5, 97.5])
conf_int_fortis = np.percentile(bs_replicates_fortis, [2.5, 97.5])
