package com.intel.oap.substrait.rel;

import com.intel.oap.substrait.type.TypeNode;
import io.substrait.ReadRel;
import io.substrait.Rel;
import io.substrait.Type;

import java.util.ArrayList;

public class ReadRelNode implements RelNode {
    private final ArrayList<TypeNode> types = new ArrayList<>();
    private final ArrayList<String> names = new ArrayList<>();
    private final ArrayList<String> paths = new ArrayList<>();

    ReadRelNode(ArrayList<TypeNode> types, ArrayList<String> names,
                ArrayList<String> paths) {
        this.types.addAll(types);
        this.names.addAll(names);
        this.paths.addAll(paths);
    }

    @Override
    public Rel toProtobuf() {
        Type.Struct.Builder structBuilder = Type.Struct.newBuilder();
        for (TypeNode typeNode : types) {
            structBuilder.addTypes(typeNode.toProtobuf());
        }
        Type.NamedStruct.Builder nStructBuilder = Type.NamedStruct.newBuilder();
        nStructBuilder.setStruct(structBuilder.build());
        for (String name : names) {
            nStructBuilder.addNames(name);
        }
        ReadRel.LocalFiles.Builder localFilesBuilder =
                ReadRel.LocalFiles.newBuilder();
        for (String path : paths) {
            ReadRel.LocalFiles.FileOrFiles.Builder fileBuiler =
                    ReadRel.LocalFiles.FileOrFiles.newBuilder();
            fileBuiler.setUriPath(path);
            localFilesBuilder.addItems(fileBuiler.build());
        }
        ReadRel.Builder readBuilder = ReadRel.newBuilder();
        readBuilder.setBaseSchema(nStructBuilder.build());
        readBuilder.setLocalFiles(localFilesBuilder.build());
        Rel.Builder builder = Rel.newBuilder();
        builder.setRead(readBuilder.build());
        return builder.build();
    }
}
