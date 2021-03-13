import Grid from '@material-ui/core/Grid';
import React, { useState } from 'react';
import UploadButton from '../components/UploadButton';
import ResultsTable from '../components/ResultsTable';
import Filters from '../components/Filters';

export default function Home() {
    const [runs, setRuns] = useState([]);
    const [params, setParams] = useState({});

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
        let params = {}

        // Potential speed up here
        data.split("\n").forEach((el, i) => {
            if (el) {
                let run = JSON.parse(el);
                run["id"] = i;
                
                runs.push(run);

                for (const [key, value] of Object.entries(run.params)) {
                    if (!params[key]) {
                        params[key] = new Set();
                    }

                    params[key].add(value);
                }
            }
        });

        setRuns(runs);
        setParams(params);
        
        // TODO:
            // save params keys (from first object)
            // go through each object and generate param values for dropdowns
    }

    // TODO: Checlist:
        // multi-dropdowns to filter data by params
            // lav dem baseret på indlæst data? I.e. tag værdierne fra daten i stedet for hardcoded. Lad være med at vise dropdown hvis der ikke er nogle værdier (løser problemet med at stance-selection har færre hyperparameters)

    return (
        <div style={{ flexGrow: 1, width: '90%', paddingLeft: "10%"}}>
                <Grid container xs style={{ paddingTop: 30 }} spacing={2}>
                    <Grid item xs>
                        <UploadButton onChange={readAndParseResultsFile}/>
                    </Grid>
                </Grid>
                <Grid container xs spacing={8}>
                    {runs.length > 0 && <Filters params={params}/>}
                </Grid>
                <Grid container xs spacing={8}>
                    {/* skal vi filter i de runs, der bliver sendt ned her? I så fald skal filter somehow op fra Filters*/}
                    {runs.length > 0 && <Grid item xs>
                        <ResultsTable runs={runs}/>
                    </Grid>}
                </Grid>
        </div>
    )
}
