import Grid from '@material-ui/core/Grid';
import React, { useState, useEffect } from 'react';
import UploadButton from '../components/UploadButton';
import ResultsTable from '../components/ResultsTable';
import Filters from '../components/Filters';

export default function Home() {
    const [runs, setRuns] = useState([]);
    const [hyperparams, setHyperparams] = useState({});
    const [filters, setFilters] = useState({});

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

                if (run.params) {
                    for (const [key, value] of Object.entries(run.params)) {
                        if (!hyperparams[key]) {
                            hyperparams[key] = new Set();
                        }
    
                        hyperparams[key].add(value);
                    }
                }
            }
        });

        setRuns(runs);
        setHyperparams(hyperparams);
        setFilters({});
    }

    const onChangeFilter = (value, key) => {
        let newFilters = {...filters};
        
        if (value.length === 0) {
            delete newFilters[key]
        } else {
            newFilters[key] = value
        }

        setFilters(newFilters);
    }

    const getFilteredRuns = () => {
        // Potential speed up here
        let filteredRuns = runs.filter(function(run) {
            for (const [key, value] of Object.entries(filters)) {
                if (!value.includes(run.params[key])) {
                    return false;
                }
            }

            return true;
        })
        
        return filteredRuns;
    }

    return (
        <div style={{ flexGrow: 1, width: '90%', paddingLeft: "10%" }}>
            {runs.length > 0 && <Grid container spacing={4} style={{ paddingTop: 20 }}>
                <Filters hyperparams={hyperparams} onChange={onChangeFilter} filters={filters}/>
            </Grid>}
            {runs.length > 0 && <Grid container spacing={8} style={{ paddingTop: 10 }}>
                <Grid item xs>
                    <ResultsTable runs={getFilteredRuns()}/>
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
