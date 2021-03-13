import Grid from '@material-ui/core/Grid';
import React, { useState } from 'react';
import UploadButton from '../components/UploadButton';
import ResultsTable from '../components/ResultsTable';
import Filters from '../components/Filters';

export default function Home() {
    const [runs, setRuns] = useState([]);
    const [hyperparams, setHyperparams] = useState({});

    // TODO: Make this async?
    const readAndParseResultsFile = event => {
        const fileReader = new FileReader();
        fileReader.readAsText(event.target.files[0])
        fileReader.onload = ev => {
            parseResultsFile(ev.target.result)
        };
    }

    const parseResultsFile = (data) => {
        let runs = [];
        let hyperparams = {}

        // Potential speed up here
        data.split("\n").forEach((el, i) => {
            if (el) {
                let run = JSON.parse(el);
                run["id"] = i;
                
                runs.push(run);

                for (const [key, value] of Object.entries(run.params)) {
                    if (!hyperparams[key]) {
                        hyperparams[key] = new Set();
                    }

                    hyperparams[key].add(value);
                }
            }
        });

        setRuns(runs);
        setHyperparams(hyperparams);
        
        // TODO:
            // save params keys (from first object)
            // go through each object and generate param values for dropdowns
    }

    // TODO: Checlist:
        // multi-dropdowns to filter data by params
            // lav dem baseret på indlæst data? I.e. tag værdierne fra daten i stedet for hardcoded. Lad være med at vise dropdown hvis der ikke er nogle værdier (løser problemet med at stance-selection har færre hyperparameters)

    return (
        <div style={{ flexGrow: 1, width: '90%', paddingLeft: "10%"}}>
                {runs.length > 0 && <Grid container spacing={8}>
                     <Filters hyperparams={hyperparams}/>
                </Grid>}
                {runs.length > 0 &&<Grid container spacing={8}>
                    {/* skal vi filter i de runs, der bliver sendt ned her? I så fald skal filter somehow op fra Filters*/}
                     <Grid item xs>
                        <ResultsTable runs={runs}/>
                    </Grid>
                </Grid>}
                <Grid container spacing={8}>
                    <Grid item xs>
                        <UploadButton onChange={readAndParseResultsFile}/>
                    </Grid>
                </Grid>
        </div>
    )
}
