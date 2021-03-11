import Button from '@material-ui/core/Button';

export default function UploadButton(props) {
    return (
        <>
            <input
                onChange={props.readResultsFile}
                style={{ display: "none" }}
                id="contained-button-file"
                type="file"
                accept=".jsonl"
            />
            <label htmlFor="contained-button-file">
                <Button variant="contained" color="primary" component="span">
                    Upload results file
                </Button>
            </label>
        </>
    )
}