import Grid from '@material-ui/core/Grid';
import React, { useState, useEffect } from 'react';
import UploadButton from '../components/UploadButton';
import { DataGrid } from '@material-ui/data-grid';

export default function Home() {
    const [runs, setRuns] = useState([]);
    const [columns, setColumns] = useState([]);
    const [rows, setRows] = useState([]);

    // TODO: Make this async?
    const readResultsFile = event => {
        const fileReader = new FileReader();
        fileReader.readAsText(event.target.files[0])
        fileReader.onload = ev => {
            parseResultsFile(ev.target.result, event.target.files[0].name)
        };
    }

    const parseResultsFile = (data, fullFileName) => {
        let tempRuns = [];
        const fileName = fullFileName.split(/[.-]/).slice(1,3).join("_"); // TODO: Make more general

        // Potential speed up here
        data.split("\n").forEach(el => {
            if (el) {
                let name = fileName;
                let run = JSON.parse(el);

                if (run.params) {
                    for (const [key, value] of Object.entries(run.params)) {
                        name = name.concat(`_${key}_${value}`);
                    }
                }
                run["name"] = name;
                
                tempRuns.push(run);
            }
        });

        setRuns(tempRuns);
        
        // TODO:
            // save params keys (from first object)
            // go through each object and generate param values for dropdowns
    }

    // update rows and columns when runs update
    useEffect(() => {
        setColumns(getColumns());
        setRows(getRows());
    }, [runs])

    const getColumns = () => {
        let cols = [];
        if (runs.length > 0) {
            cols.push({field: "name", flex: 0.5});
            Object.keys(runs[0].results).forEach(key => {
                cols.push({field: key, flex: 1});
            })
        }
        return cols;
    }
        
    const getRows = () => {
        let rows = [];
        if (runs.length > 0) {
            for (let i = 0; i < runs.length; i++) {
                let row = {id: i, name: runs[i]["name"]};

                for (const [key, value] of Object.entries(runs[i].results)) {
                    row[key] = value;
                }

                rows.push(row);
            }
        }
        return rows;
    }

    // TODO: Checlist:
        // if data, show row with table (datagrid) 
            // pages i table?
            // Standard sort efter f1
        // row with multi-dropdowns to filter data (skal egentlig være over table)
        // lav dem baseret på indlæst data? I.e. tag værdierne fra daten i stedet for hardcoded. Lad være med at vise dropdown hvis der ikke er nogle værdier (løser problemet med at stance-selection har færre hyperparameters)
        // select rows (in table) to compare in another table (which is only shown if at least two datapoints are selected)

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
                    <UploadButton readResultsFile={readResultsFile}/>
                </Grid>
                <Grid item xs>
                    {"Dropdowns"}
                </Grid>
                <Grid item xs style={{width: '80%'}}>
                    {<DataGrid 
                        rows={rows}
                        columns={columns} 
                        pageSize={15} 
                        checkboxSelection
                        autoHeight={true}
                    />}
                </Grid>
            </Grid>
        </div>
    )
}
