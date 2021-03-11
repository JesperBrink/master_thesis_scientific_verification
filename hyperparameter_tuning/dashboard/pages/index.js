import Grid from '@material-ui/core/Grid';
import React, { useState, useEffect } from 'react';
import UploadButton from '../components/UploadButton';

export default function Home() {
    const [results, setResults] = useState([]);
    const [resultsKeys, setResultsKeys] = useState([]);

    // TODO: Make this async?
    const readResultsFile = event => {
        const fileReader = new FileReader();
        fileReader.readAsText(event.target.files[0])
        fileReader.onload = ev => {
            parseResultsFile(ev.target.result, event.target.files[0].name)
        };
    }

    const parseResultsFile = (data, fullFileName) => {
        let hyperparameterRuns = [];
        const fileName = fullFileName.split(/[.-]/).slice(1,3).join("_");

        // Potential speed up here
        data.split("\n").forEach(el => {
            if (el) {
                let name = fileName;
                let run = JSON.parse(el);
                console.log(run.params)

                for (const [key, value] of Object.entries(run.params)) {
                    name = name.concat(`_${key}_${value}`);
                }
                run["name"] = name;
                
                hyperparameterRuns.push(run);
            }
        });

        setResults(hyperparameterRuns);
        setResultsKeys(Object.keys(hyperparameterRuns[0].results));
        
        // TODO:
            // save params keys (from first object)
            // go through each object and generate param values for dropdowns
    }

    // useEffect(() => {
    //     console.log(results);
    // }, [results])

    // TODO: Checlist:
        // if data, show row with table (datagrid) 
            // pages i table?
            // Standard sort efter f1
        // row with multi-dropdowns to filter data (skal egentlig være over table)
        // lav dem baseret på indlæst data? I.e. tag værdierne fra daten i stedet for hardcoded. Lad være med at vise dropdown hvis der ikke er nogle værdier (løser problemet med at stance-selection har færre hyperparameters)
        // select rows (in table) to compare in another table (which is only shown if at least two datapoints are selected)

    return (
        <div >
            <Grid 
                container 
                spacing={2}
                direction="column"
                alignContent="center"
                style={{ flexGrow: 1, paddingTop: 30 }}
            >
                <Grid item xs>
                    <UploadButton readResultsFile={readResultsFile}/>
                </Grid>
                <Grid item xs>
                    {"Dropdowns"}
                </Grid>
                <Grid item xs>
                    {"Table"}
                </Grid>
            </Grid>
        </div>
    )
}
