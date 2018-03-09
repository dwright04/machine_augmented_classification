<!doctype html>
<meta charset="utf8">
<script src="http://distill.pub/template.v1.js"></script>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css" integrity="sha384-wITovz90syo1dJWVh32uuETPVEtGigN07tkttEqPv+uR2SE/mbQcG7ATL28aI9H0" crossorigin="anonymous">

<script type="text/front-matter">
  title: Augementing Image Classification in Citizen Scientist Projects with Artificial Intelligence
  published: Feb 06, 2018
  authors:
  - Darryl Wright:
  - Michael Laraia:
  - Lucy Fortson:
  - Chris Lintott:
  affiliations:
  - University of Minnesota:
  - University of Minnesota:
  - University of Minnesota:
  - University of Oxford:
</script>

<dt-article>
  <script type="text/article"></script>
  <h1>Augementing Image Classification in Citizen Scientist Projects with Artificial Intelligence</h1>
  <h2></h2>
  <dt-byline></dt-byline>
  <h2>Introduction</h2>
    <p>
    </p>
    <h3>Related Work</h3>
      <p>
      </p>
  <h2>Data and Method</h2>
    <h3>Setting the Goal</h3>
    <h3>Our Method</h3>
      <p>
        <ul style="list-style-type:circle">
          <li>Unsupervised Deep Embedding Clustering</li>
          <li>Citizen Science</li>
            <ul style="list-style-type:square">
              <li>Supernova Hunters</li>
              <li>Muon Hunter</li>
            </ul>
          <li>Simulating Volunteer Classifications</li>
            <ul style="list-style-type:square">
              <li>Perfect Classifiers</li>
              <li>Adding Random Noise</li>
              <li>Correlated Noise from Volunteers</li>
            </ul>
          <li>Selecting Queries</li>
          <li>Mapping Clusters to Labels</li>
          <li>Building the Training Set</li>
            <ul style="list-style-type:square">
              <li>Concatentation of Labelled Examples</li>
            </ul>
        </ul>
      </p>
    <h3>Datasets</h3>
      <p>
        <ul style="list-style-type:circle">
          <li>MNIST</li>
          <li>CIFAR-10</li>
          <li>Supernova Hunters</li>
          <li>Muon Hunter</li>
        </ul>
      </p>
    <h3>Evaluation and Metrics</h3>
    <h3>Implementation</h3>
  <h2>Experimental Results</h2>
    <h3>Unsupervised Clustering</h3>
      <p>How does the completely unsupervised approach perform across the data sets for which we have a validation set? So performance for MNIST, CIFAR-10 and Supernova Hunters.  Muon Hunters is considered here as we have not gold standard data set, it is "held out" as a real test example of the approach.
      </p>
      <p>What are the implications for the different data sets.
      </p>
      <p>Tables
        <ul style="list-style-type:square">
          <li>Showing performance for each of the 3 data sets measured on respective validation sets.</li>
        </ul>
      </p>
      <p>Plots
        <ul style="list-style-type:square">
          <li>Examples of e.g. 5 clusters from each of the 3 data sets</li>
        </ul>
      </p>
    <h3></h3>
      <p>
      </p>
    <h3>Ablation Studies</h3>
      <p>In order to better understand our approach we perform tests to disentangle the contribution to the performance of several implementation choices.
        <ul style="list-style-type:circle">
          <li>Batch Size for Model Updates</li>
            <p>Plots
              <ul style="list-style-type:square">
                <li>3 plots showing learning curves with batches of different sizes.</li>
              </ul>
            </p>
          <li>Number of Clusters</li>
            <p>Tables
              <ul style="list-style-type:square">
                <li>Final performance of the approach on all 3 data sets with 4 different cluster sizes e.g. n_classes, 10, 100 and 1000</li>
              </ul>
            </p>
        </ul>
      </p>
    <h3>Test Set Results</h3>
      <p>Tables
        <ul style="list-style-type:square">
          <li>Showing performance for each of the 3 data sets measured on respective test sets.</li>
        </ul>
      </p>
      <p>Plots
        <ul style="list-style-type:square">
          <li>Examples of e.g. 5 clusters from each of the 3 test sets</li>
        </ul>
      </p>
    <h3>Real-World Application</h3>
      <p>
        <ul style="list-style-type:circle">
          <li></li>
        </ul>
      </p>
  <h2>Discussion</h2>
  <h2>Conclusions</h2>
</dt-article>
