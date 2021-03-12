import Grid from '@material-ui/core/Grid';
import React, { useState } from 'react';
import UploadButton from '../components/UploadButton';
import ResultsTable from '../components/ResultsTable';
import Autocomplete from '@material-ui/lab/Autocomplete';

export default function Home() {
    const [runs, setRuns] = useState([]);

    // TODO: Make this async?
    const readAndParseResultsFile = event => {
        const fileReader = new FileReader();
        fileReader.readAsText(event.target.files[0])
        fileReader.onload = ev => {
            parseResultsFile(ev.target.result)
        };
    }

    const parseResultsFile = (data) => {
        let tempRuns = [];

        // Potential speed up here
        data.split("\n").forEach((el, i) => {
            if (el) {
                let run = JSON.parse(el);
                run["id"] = i;
                
                tempRuns.push(run);
            }
        });

        setRuns(tempRuns);
        
        // TODO:
            // save params keys (from first object)
            // go through each object and generate param values for dropdowns
    }

    // TODO: Checlist:
        // multi-dropdowns to filter data by params
            // lav dem baseret på indlæst data? I.e. tag værdierne fra daten i stedet for hardcoded. Lad være med at vise dropdown hvis der ikke er nogle værdier (løser problemet med at stance-selection har færre hyperparameters)

    return (
        <div style={{ height: '100%' }}>
            <Grid 
                container 
                spacing={2}
                direction="column"
                alignContent="center"
                style={{ flexGrow: 1, paddingTop: 30 }}
            >
                <Grid item xs>
                    <UploadButton onChange={readAndParseResultsFile}/>
                </Grid>
                <Grid item xs>
                    {runs.length > 0 && <span>Dropdowns</span>}
                </Grid>
                <Grid item xs style={{width: '80%'}}>
                    {runs.length > 0 && <ResultsTable runs={runs}/>}
                </Grid>
            </Grid>
        </div>
    )
}
