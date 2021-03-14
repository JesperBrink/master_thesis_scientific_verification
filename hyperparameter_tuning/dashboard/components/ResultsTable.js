import React, { useState } from 'react';
import { DataGrid } from '@material-ui/data-grid';
import Tooltip from '@material-ui/core/Tooltip';

export default function ResultsTable({ runs = [] }) {

    const getDataRows = () => {
        let rows = [];
        // Create rows from runs
        if (runs.length > 0) {
            for (let i = 0; i < runs.length; i++) {
                let row = { id: runs[i].id, params: runs[i].params };

                for (const [key, value] of Object.entries(runs[i].results)) {
                    row[key] = value;
                }

                rows.push(row);
            }
        }
        return rows;
    }

    const getDataColumns = () => {
        let cols = [];
        // Create columns from runs
        if (runs.length > 0) {
            cols.push({
                field: "id",
                flex: 0.5,
                renderCell: (params) => (
                    <Tooltip title={formatParamsTooltip(params.row.params)} interactive>
                        <span>{params.row.id}</span>
                    </Tooltip>
                )
            });

            Object.keys(runs[0].results).forEach(key => {
                cols.push({
                    field: key,
                    flex: 1,
                    renderCell: (params) => (
                        <Tooltip title={<span style={{ fontSize: 16 }}>{params.row[key]}</span>}>
                            <span>{parseFloat(params.row[key]).toFixed(5)}</span>
                        </Tooltip>
                    )
                }); // add headerName if we want aliases
            })
        }
        return cols;
    }

    const formatParamsTooltip = (params) => {
        let tooltip = []

        if (params) {
            for (const [key, value] of Object.entries(params)) {
                if (key === "id") {
                    continue;
                }

                tooltip.push(<div key={key} style={{ fontSize: 16 }}>{key + ": " + value}</div>)
            }
        }

        return tooltip;
    }

    return (
        <>
            {runs.length > 0 && <DataGrid
                rows={getDataRows()}
                columns={getDataColumns()}
                pageSize={15}
                rowHeight={46}
                autoHeight={true}
            />}
        </>
    )
}