import React, { useState, useEffect } from 'react';
import { DataGrid } from '@material-ui/data-grid';
import Tooltip from '@material-ui/core/Tooltip';

export default function ResultsTable(props) {
    const [columns, setColumns] = useState([]);
    const [rows, setRows] = useState([]);

    // update rows and columns when runs updates
    useEffect(() => {
        updateColumns();
        updateRows();
    }, [props.runs])

    const updateColumns = () => {
        let cols = [];
        // Create columns from runs
        if (props.runs.length > 0) {
            cols.push({
                field: "id", 
                flex: 0.5,
                renderCell: (params) =>  (
                    <Tooltip title={formatParamsTooltip(params.row.id)} interactive>
                        <span>{params.row.id}</span>
                    </Tooltip>
                )
            });
            
            Object.keys(props.runs[0].results).forEach(key => {
                cols.push({field: key, 
                    flex: 1, 
                    renderCell: (params) =>  (
                        <Tooltip title={<h2>{params.row[key]}</h2>}>
                            <span>{parseFloat(params.row[key]).toFixed(5)}</span>
                        </Tooltip>
                    )
                }); // add headerName if we want aliases
            })
        }
        setColumns(cols);
    }
        
    const updateRows = () => {
        let rows = [];
        // Create rows from runs
        if (props.runs.length > 0) {
            for (let i = 0; i < props.runs.length; i++) {
                let row = {id: props.runs[i].id};

                for (const [key, value] of Object.entries(props.runs[i].results)) {
                    row[key] = value;
                }

                rows.push(row);
            }
        }
        setRows(rows);
    }

    const formatParamsTooltip = (id) => {
        let tooltip = []

        if (props.runs[id].params) {
            for (const [key, value] of Object.entries(props.runs[id].params)) {
                if (key === "id") {
                    continue;
                }
                
                tooltip.push(<h2 key={key}>{key + ": " + value + "\n"}</h2>)
            }
        }

        return tooltip;
    }

    return (
        <DataGrid 
            rows={rows}
            columns={columns} 
            pageSize={14} 
            autoHeight={true}
        />
    )
}