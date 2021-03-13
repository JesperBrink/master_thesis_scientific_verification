import React from 'react';
import { DataGrid } from '@material-ui/data-grid';
import Tooltip from '@material-ui/core/Tooltip';

export default function ResultsTable({ runs = [] }) {
    const getColumns = () => {
        let cols = [];
        // Create columns from runs
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
                }); // add headerName if we want aliases
            })
        }
        return cols;
    }
        
    const getRows = () => {
        let rows = [];
        // Create rows from runs
        if (runs.length > 0) {
            for (let i = 0; i < runs.length; i++) {
                let row = {id: runs[i].id};

                for (const [key, value] of Object.entries(runs[i].results)) {
                    row[key] = value;
                }

                rows.push(row);
            }
        }
        return rows;
    }

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

    return (
        <>
            {runs.length > 0 && <DataGrid 
                rows={getRows()}
                columns={getColumns()} 
                pageSize={14} 
                autoHeight={true}
            />}
        </>
    )
}