import Grid from '@material-ui/core/Grid';
import React, { useState, useEffect } from 'react';
import UploadButton from '../components/UploadButton';
import { DataGrid } from '@material-ui/data-grid';
import Tooltip from '@material-ui/core/Tooltip';

export default function Home() {
    const [runs, setRuns] = useState([]);
    const [columns, setColumns] = useState([]);
    const [rows, setRows] = useState([]);

    // TODO: Make this async?
    const readResultsFile = event => {
        const fileReader = new FileReader();
        fileReader.readAsText(event.target.files[0])
        fileReader.onload = ev => {
            parseResultsFile(ev.target.result)
        };
    }

    // lav tooltip på hele row med ordentligt formaterede hyper parameters

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

    // update rows and columns when runs updates
    useEffect(() => {
        updateColumns();
        updateRows();
    }, [runs])

    const formatParamsTooltip = (id) => {
        let tooltip = []

        if (runs[id].params) {
            for (const [key, value] of Object.entries(runs[id].params)) {
                if (key === "id") {
                    continue;
                }
                
                tooltip.push(<h2 key={key}>{key + ": " + value + "\n"}</h2>)
            }
        }

        return tooltip;
    }

    const updateColumns = () => {
        let cols = [];
        if (runs.length > 0) {

            cols.push({
                field: "id", 
                flex: 0.5,
                renderCell: (params) =>  (
                    <Tooltip title={formatParamsTooltip(params.row.id)} interactive>
                        <span>{params.row.id}</span>
                    </Tooltip>
                )
            });
            
            Object.keys(runs[0].results).forEach(key => {
                cols.push({field: key, 
                    flex: 1, 
                    renderCell: (params) =>  (
                        <Tooltip title={<h2>{params.row[key]}</h2>}>
                            <span>{parseFloat(params.row[key]).toFixed(5)}</span>
                        </Tooltip>
                    )
                }); // add headerName if we want aliases, remove type if we want full values (also ruins result filtering)
            })
        }
        setColumns(cols);
    }
        
    const updateRows = () => {
        let rows = [];
        if (runs.length > 0) {
            // Create rows from runs
            for (let i = 0; i < runs.length; i++) {
                let row = {id: runs[i].id};

                for (const [key, value] of Object.entries(runs[i].results)) {
                    row[key] = value;
                }

                rows.push(row);
            }
        }
        setRows(rows);
    }

    // TODO: Checlist:
        // multi-dropdowns to filter data by params
        // lav dem baseret på indlæst data? I.e. tag værdierne fra daten i stedet for hardcoded. Lad være med at vise dropdown hvis der ikke er nogle værdier (løser problemet med at stance-selection har færre hyperparameters)
        // MAYBE: select rows (in table) to compare in another table (which is only shown if at least two datapoints are selected)

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
                    {runs.length > 0 && <span>Dropdowns</span>}
                </Grid>
                <Grid item xs style={{width: '80%'}}>
                    {runs.length > 0 &&<DataGrid 
                        rows={rows}
                        columns={columns} 
                        pageSize={14} 
                        autoHeight={true}
                    />}
                </Grid>
            </Grid>
        </div>
    )
}
