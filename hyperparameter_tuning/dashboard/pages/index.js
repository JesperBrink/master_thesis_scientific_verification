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
    }

    // TODO: Checlist:
    // make dropdowns filter data

    return (
        <div style={{ flexGrow: 1, width: '90%', paddingLeft: "10%" }}>
            {runs.length > 0 && <Grid container spacing={4}>
                <Filters hyperparams={hyperparams} />
            </Grid>}
            {runs.length > 0 && <Grid container spacing={8} style={{ paddingTop: 10 }}>
                {/* skal vi filter i de runs, der bliver sendt ned her? I så fald skal filter somehow op fra Filters*/}
                <Grid item xs>
                    <ResultsTable runs={runs} />
                </Grid>
            </Grid>}
            <Grid container spacing={8}>
                <Grid item xs>
                    <UploadButton onChange={readAndParseResultsFile} />
                </Grid>
            </Grid>
        </div>
    )
}
